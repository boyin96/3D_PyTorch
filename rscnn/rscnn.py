import torch
import torch.nn as nn

from pointnet2.pointnet2 import PointNetSetAbstraction


class Net(nn.Module):
    def __init__(self, num_classes, normal_channel=True):
        super().__init__()

        self.num_classes = num_classes
        self.normal_channel = normal_channel
        input_channels = 6 if self.normal_channel else 3
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=512,
                radius=0.23,
                nsample=32,
                in_channel=input_channels,
                mlp=[64, 64, 128],
                group_all=False
            )
        )
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=128,
                radius=0.4,
                nsample=64,
                in_channel=128 + 3,
                mlp=[128, 128, 256],
                group_all=False
            )
        )
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=None,
                radius=None,
                nsample=None,
                in_channel=256 + 3,
                mlp=[256, 512, 1024],
                group_all=True
            )
        )

        self.FC_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=0.5),
            nn.Linear(256, self.num_classes)
        )

    def forward(self, pointcloud):
        xyz = pointcloud[:, :3, :]
        if self.normal_channel:
            features = pointcloud[:, 3:, :]
        else:
            features = None
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return self.FC_layer(features.squeeze(-1))


if __name__ == '__main__':
    sim_data = torch.rand(32, 3, 2500)
    seg = Net(12, False)
    out = seg(sim_data)
    print("seg: ", out.size())
