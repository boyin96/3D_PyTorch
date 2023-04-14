import argparse
import os
import random
import torch.nn.functional as F
import torch.utils.data

import utilities as ut
from datasets import MyDataset
from pointnet2 import PointNetClsSSG

# Set parameters.
parser = argparse.ArgumentParser("training")
parser.add_argument("--batch_size", type=int, default=32, help="input batch size")
parser.add_argument("--num_points", type=int, default=2500, help="input points size")
parser.add_argument("--workers", type=int, default=0, help="number of data loading workers")
parser.add_argument("--num_epoch", type=int, default=200, help="number of epochs to train")
parser.add_argument("--out_folder", type=str, default="cls", help="output folder")
parser.add_argument("--model_path", type=str, default="", help="model path")
parser.add_argument("--use_normals", type=bool, default=True, help="use normals")
parser.add_argument("--dataset_path", type=str, default="shapenetcore_partanno_segmentation_benchmark_v0",
                    help="dataset path")
parser.add_argument("--dataset_type", type=str, default="prediction", help="dataset type")
parser.add_argument("--feature_transform_regular", type=bool, default=False, help="use feature transform")
opt = parser.parse_args()

# Set CPU seed.
opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Set dataset.
if opt.dataset_type == "prediction":
    train_dataset = MyDataset(
        root=opt.dataset_path,
        classification=True,
        train=True,
        npoints=opt.num_points,
        data_aug=True
    )
    test_dataset = MyDataset(
        root=opt.dataset_path,
        classification=True,
        train=False,
        npoints=opt.num_points,
        data_aug=False
    )
else:
    train_dataset, test_dataset = None, None
    exit("Wrong dataset type!")

# Set dataloader.
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.workers)
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=int(opt.workers)
)

num_classes = len(train_dataset.classes)

try:
    os.makedirs(opt.out_folder)
except OSError:
    pass

# Load model.
classifier = PointNetClsSSG(num_classes, normal_channel=opt.use_normals)

if opt.model_path != "":
    classifier.load_state_dict(torch.load(opt.model_path))

optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
if torch.cuda.is_available():
    classifier.cuda()

# Train model.
num_batch = len(train_dataset) / opt.batch_size

for epoch in range(opt.num_epoch):
    classifier = classifier.train()
    for i, data in enumerate(train_dataloader, 0):
        optimizer.zero_grad()

        points, target = data
        target = target[:, 0]
        points = points.detach().numpy()
        points = ut.random_point_dropout(points)
        points[:, :, 0:3] = ut.random_scale_point_cloud(points[:, :, 0:3])
        points[:, :, 0:3] = ut.shift_point_cloud(points[:, :, 0:3])
        points = torch.from_numpy(points)
        points = points.transpose(2, 1)

        if torch.cuda.is_available():
            points, target = points.cuda(), target.cuda()
        pred, _ = classifier(points)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.detach().max(1)[1]
        correct = pred_choice.eq(target.detach()).cpu().sum()
        print("[%d: %d/%d] train loss: %f accuracy: %f" % (
            epoch, i, num_batch, loss.item(), correct.item() / float(opt.batch_size)))

        if i % 10 == 0:
            with torch.no_grad():
                classifier = classifier.eval()
                j, data = next(enumerate(test_dataloader, 0))
                points, target = data
                target = target[:, 0]
                points = points.transpose(2, 1)
                if torch.cuda.is_available():
                    points, target = points.cuda(), target.cuda()
                pred, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.detach().max(1)[1]
                correct = pred_choice.eq(target.detach()).cpu().sum()
                print("[%d: %d/%d] %s loss: %f accuracy: %f" % (
                    epoch, i, num_batch, "test", loss.item(), correct.item() / float(opt.batch_size)))

    torch.save(classifier.state_dict(), "%s/cls_model_%d.pth" % (opt.out_folder, epoch))
