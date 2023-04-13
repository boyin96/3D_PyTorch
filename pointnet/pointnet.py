import torch
import torch.nn as nn
import torch.nn.functional as F


class STN3d(nn.Module):
    """
    Input transform.
    """

    def __init__(self, num_points=2500):
        super(STN3d, self).__init__()

        self.num_points = num_points

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.mp1 = nn.MaxPool1d(self.num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3).flatten().view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)

        # shape of x: [batchsize, 3, 3]
        return x


class STNkd(nn.Module):
    """
    Feature transform.
    """

    def __init__(self, num_points=2500, k=64):
        super(STNkd, self).__init__()

        self.num_points = num_points
        self.k = k

        self.conv1 = nn.Conv1d(self.k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.mp1 = nn.MaxPool1d(self.num_points)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.mp1(x)
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k).flatten().view(1, self.k * self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden

        # shape of x: [batchsize, self.k, self.k]
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    """
    Obtain the features of point cloud data.
    """

    def __init__(self, num_points=2500, k=64, global_feat=True, feature_trans=False):
        super(PointNetfeat, self).__init__()

        self.num_points = num_points
        self.k = k
        self.global_feat = global_feat
        self.feature_trans = feature_trans

        self.stn = STN3d(num_points=self.num_points)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.mp1 = nn.MaxPool1d(self.num_points)

        if self.feature_trans:
            self.fstn = STNkd(num_points=self.num_points, k=self.k)

    def forward(self, x):
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_trans:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        point_feature = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
            return torch.cat([x, point_feature], 1), trans, trans_feat


class PointNetCls(nn.Module):
    def __init__(self, num_points=2500, k=2, feature_trans=False):
        super(PointNetCls, self).__init__()

        self.num_points = num_points
        self.k = k
        self.feature_trans = feature_trans

        self.feat = PointNetfeat(self.num_points, global_feat=True, feature_trans=self.feature_trans)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=-1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, num_points=2500, k=2, feature_trans=False):
        super(PointNetDenseCls, self).__init__()

        self.num_points = num_points
        self.k = k
        self.feature_trans = feature_trans

        self.feat = PointNetfeat(self.num_points, global_feat=False, feature_trans=self.feature_trans)

        self.conv1 = nn.Conv1d(1088, 512, 1)
        self.conv2 = nn.Conv1d(512, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        x = x.view(batchsize, self.num_points, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d).unsqueeze(0)
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.linalg.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


# Test point network
if __name__ == '__main__':
    sim_data = torch.rand(32, 3, 2500)
    sim_data_64d = torch.rand(32, 64, 2500)

    # tran = STN3d()
    # out = tran(sim_data)
    # print("stn: ", out.size())
    # # print('loss', feature_transform_regularizer(out))

    # tran = STNkd(k=64)
    # out = tran(sim_data_64d)
    # print("stn64d: ", out.size())
    # # print('loss', feature_transform_regularizer(out))

    # pointfea = PointNetfeat(global_feat=True)
    # out, _, _ = pointfea(sim_data)
    # print("global feat: ", out.size())
    #
    # pointfea = PointNetfeat(global_feat=False)
    # out, _, _ = pointfea(sim_data)
    # print("point feat: ", out.size())

    # cls = PointNetCls(k=5)
    # out, _, _ = cls(sim_data)
    # print("class: ", out.size())

    # seg = PointNetDenseCls(k=3)
    # out, _ = seg(sim_data)
    # print("seg: ", out.size())

    # stn: torch.Size([32, 3, 3])
    # global feat:  torch.Size([32, 1024])
    # point feat: torch.Size([32, 1088, 2500])
    # class:  torch.Size([32, 5])
    # seg: torch.Size([32, 2500, 3])
