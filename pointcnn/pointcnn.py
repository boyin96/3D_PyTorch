import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sklearn.neighbors import NearestNeighbors


def knn_indices_func_cpu(rep_pts, pts, K, D):
    rep_pts = rep_pts.data.numpy()
    pts = pts.data.numpy()
    region_idx = []

    for n, p in enumerate(rep_pts):
        P_particular = pts[n]
        nbrs = NearestNeighbors(n_neighbors=D * K + 1, algorithm="ball_tree").fit(P_particular)
        indices = nbrs.kneighbors(p)[1]
        region_idx.append(indices[:, 1::D])

    region_idx = torch.from_numpy(np.stack(region_idx, axis=0))
    return region_idx


def knn_indices_func_gpu(rep_pts, pts, k):
    region_idx = []

    for n, qry in enumerate(rep_pts):
        ref = pts[n]
        n, d = ref.size()
        m, d = qry.size()
        mref = ref.expand(m, n, d)
        mqry = qry.expand(n, m, d).transpose(0, 1)
        dist2 = torch.sum((mqry - mref) ** 2, 2).squeeze()
        _, inds = torch.topk(dist2, k * d + 1, dim=1, largest=False)
        region_idx.append(inds[:, 1::d])

    region_idx = torch.stack(region_idx, dim=0)
    return region_idx


def EndChannels(f):
    class WrappedLayer(nn.Module):
        def __init__(self):
            super(WrappedLayer, self).__init__()

            self.f = f

        def forward(self, x):
            x = x.permute(0, 3, 1, 2)
            x = self.f(x)
            x = x.permute(0, 2, 3, 1)
            return x

    return WrappedLayer()


class Dense(nn.Module):
    def __init__(self, in_features, out_features, drop_rate=0.0, activation=True):
        super(Dense, self).__init__()

        self.linear = nn.Linear(in_features, out_features)
        self.activation = activation
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else None

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = nn.ReLU(x)
        if self.drop:
            x = self.drop(x)
        return x


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, with_bn=True, activation=True):
        super(Conv, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=not with_bn)
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9) if with_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = nn.ReLU(x)
        if self.bn:
            x = self.bn(x)
        return x


class SepConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, depth_multiplier=1, with_bn=True, activation=True):
        super(SepConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * depth_multiplier, kernel_size, groups=in_channels),
            nn.Conv2d(in_channels * depth_multiplier, out_channels, 1, bias=not with_bn)
        )
        self.activation = activation
        self.bn = nn.BatchNorm2d(out_channels, momentum=0.9) if with_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.activation:
            x = nn.ReLU(x)
        if self.bn:
            x = self.bn(x)
        return x


class XConv(nn.Module):
    def __init__(self, C_in, C_out, dims, K, P, C_mid, depth_multiplier):
        super(XConv, self).__init__()

        self.P = P
        self.dense1 = Dense(dims, C_mid)
        self.dense2 = Dense(C_mid, C_mid)
        self.x_trans = nn.Sequential(
            EndChannels(Conv(
                in_channels=dims,
                out_channels=K * K,
                kernel_size=(1, K),
                with_bn=False
            )),
            Dense(K * K, K * K),
            Dense(K * K, K * K, activation=False)
        )
        self.end_conv = EndChannels(SepConv(
            in_channels=C_mid + C_in,
            out_channels=C_out,
            kernel_size=(1, K),
            depth_multiplier=depth_multiplier
        ))

    def forward(self, x):
        rep_pt, pts, fts = x
        N = len(pts)
        P = rep_pt.size()[1]  # (N, P, K, dims)
        p_center = torch.unsqueeze(rep_pt, dim=2)  # (N, P, 1, dims)
        pts_local = pts - p_center  # (N, P, K, dims)

        # Individually lift each point into C_mid space.
        fts_lifted0 = self.dense1(pts_local)
        fts_lifted = self.dense2(fts_lifted0)  # (N, P, K, C_mid)

        if fts is None:
            fts_cat = fts_lifted
        else:
            fts_cat = torch.cat((fts_lifted, fts), -1)  # (N, P, K, C_mid + C_in)

        # Learn the (N, K, K) X-transformation matrix.
        X_shape = (N, P, self.K, self.K)
        X = self.x_trans(pts_local)
        X = X.view(*X_shape)

        # Weight and permute fts_cat with the learned X.
        fts_X = torch.matmul(X, fts_cat)
        fts_p = self.end_conv(fts_X).squeeze(dim=2)
        return fts_p


class PointCNN(nn.Module):
    def __init__(self, C_in, C_out, dims, K, D, P, r_indices_func):
        super(PointCNN, self).__init__()

        C_mid = C_out // 2 if C_in == 0 else C_out // 4

        if C_in == 0:
            depth_multiplier = 1
        else:
            depth_multiplier = min(int(np.ceil(C_out / C_in)), 4)

        self.r_indices_func = lambda rep_pts, pts: r_indices_func(rep_pts, pts, K, D)
        self.dense = Dense(C_in, C_out // 2) if C_in != 0 else None
        self.x_conv = XConv(C_out // 2 if C_in != 0 else C_in, C_out, dims, K, P, C_mid, depth_multiplier)
        self.D = D

    @staticmethod
    def select_region(pts, pts_idx):
        regions = torch.stack([
            pts[n][idx, :] for n, idx in enumerate(torch.unbind(pts_idx, dim=0))
        ], dim=0)
        return regions

    def forward(self, x):
        rep_pts, pts, fts = x
        fts = self.dense(fts) if fts is not None else fts

        # This step takes ~97% of the time. Prime target for optimization: KNN on GPU.
        pts_idx = self.r_indices_func(rep_pts.cpu(), pts.cpu())

        pts_regional = self.select_region(pts, pts_idx)
        fts_regional = self.select_region(fts, pts_idx) if fts is not None else fts
        fts_p = self.x_conv((rep_pts, pts_regional, fts_regional))

        return fts_p


class RandPointCNN(nn.Module):
    def __init__(self, C_in: int, C_out: int, dims: int, K: int, D: int, P: int,
                 r_indices_func):

        super(RandPointCNN, self).__init__()
        self.pointcnn = PointCNN(C_in, C_out, dims, K, D, P, r_indices_func)
        self.P = P

    def forward(self, x):
        pts, fts = x
        if 0 < self.P < pts.size()[1]:
            # Select random set of indices of subsampled points.
            idx = np.random.choice(pts.size()[1], self.P, replace=False).tolist()
            rep_pts = pts[:, idx, :]
        else:
            # All input points are representative points.
            rep_pts = pts
        rep_pts_fts = self.pointcnn((rep_pts, pts, fts))
        return rep_pts, rep_pts_fts


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.pcnn1 = RandPointCNN(3, 32, 3, 8, 1, -1, knn_indices_func_cpu)
        self.pcnn2 = nn.Sequential(
            RandPointCNN(32, 64, 3, 8, 2, -1, knn_indices_func_cpu),
            RandPointCNN(64, 96, 3, 8, 4, -1, knn_indices_func_cpu),
            RandPointCNN(96, 128, 3, 12, 4, 120, knn_indices_func_cpu),
            RandPointCNN(128, 160, 3, 12, 6, 120, knn_indices_func_cpu)
        )
        self.fcn = nn.Sequential(
            Dense(160, 128),
            Dense(128, 64, drop_rate=0.5),
            Dense(64, 10, activation=False)
        )

    def forward(self, x):
        x = self.pcnn1(x)
        x = self.pcnn2(x)[1]
        x = self.fcn(x)
        return F.log_softmax(x, dim=-1)
