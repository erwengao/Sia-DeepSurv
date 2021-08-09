from project.shandong.survival.model.surv_net import generate_model
from project.shandong.survival.model.surv_net import NegativeLogLikelihood
from project.shandong.survival.model.Surv_dataset import SurvFolder
from project.shandong.survival.model.surv_utils import c_index, adjust_learning_rate
import torchio.transforms as transforms
import torchio as tio
from torch.utils.data.sampler import WeightedRandomSampler, SequentialSampler, SubsetRandomSampler
import torch
import torch.nn as nn
import numpy as np
import os
import datetime
from torch.nn import init
import torch.optim as optim

t = datetime.date.today()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0]
data_transforms = {
    'train': transforms.Compose([
        tio.Resample(1),
        # transforms.ToCanonical(p=0.5, copy=False),
        transforms.RandomFlip(flip_probability=0.5),
        transforms.RandomNoise(),
        transforms.ZNormalization()
    ]),
    'test': transforms.Compose([
        tio.Resample(1),
        transforms.ZNormalization()
    ]),
}
data_dir = r'/data/gaohan/deeplearning/project/shandong/survival/train_test/3d_split_1'
DataSets = {x: SurvFolder(os.path.join(data_dir, x), data_transforms[x])
            for x in ['train', 'test']}
# DataSets = {x: SurvFolder(os.path.join(data_dir, x))
#              for x in ['train', 'test']}
sameler_weight = [1 if DataSets['train'].imgs[i][2] == 1 else 1 for i in range(len(DataSets['train'].imgs))]
sampler = {
    # 'train': WeightedRandomSampler(sameler_weight, num_samples=2000, replacement=True),
    'train': SubsetRandomSampler(
        np.random.choice(range(len(DataSets['train'])), len(DataSets['train']), replace=False)),
    'test': SubsetRandomSampler(
        np.random.choice(range(len(DataSets['test'])), len(DataSets['test']), replace=False))
}
SurDataLoaders = {x: torch.utils.data.DataLoader(DataSets[x], batch_size=128 * len(device_ids), sampler=sampler[x])
                  for x in ['train', 'test']}
# SurDataLoaders = {x: torch.utils.data.DataLoader(DataSets[x], batch_size=64, shuffle=True)
#                  for x in ['train', 'test']}
dataset_sizes = {x: len(SurDataLoaders[x].sampler) for x in ['train', 'test']}


def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


model = generate_model(10, pretrain=True)
# model.apply(weigth_init)
model.to(device)
criterion = NegativeLogLikelihood(model, device=device).to(device)
optimizer = eval('optim.{}'.format
                 ('SGD'))(model.parameters(), lr=1e-3)
if len(device_ids) > 1:
    print("Let's use", len(device_ids), "GPUs!")
    model = nn.DataParallel(model, device_ids=device_ids)
num_epochs = 50
patience = 50
best_c_index = 0
best_epoch = 0
flag = 0

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 100)
    lr = adjust_learning_rate(optimizer, epoch, 1e-3, 1.636e-4)
    total_train_loss = 0
    total_train_c = 0
    model.train()
    for img1, img2, e, y, ids in SurDataLoaders['train']:
        model.train()
        img1 = img1.float().to(device)
        img2 = img2.float().to(device)
        e = e.to(device)
        y = y.to(device)
        risk_pred = model(img1, img2)
        train_loss = criterion(risk_pred, y, e, model)
        train_c = c_index(-risk_pred, y, e)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        total_train_loss += train_loss / (len(SurDataLoaders['train']))
        total_train_c += train_c / len(SurDataLoaders['train'])

    total_val_loss = 0
    total_val_c = 0
    model.eval()
    for img1, img2, e, y, ids in SurDataLoaders['test']:
        with torch.no_grad():
            img1 = img1.float().to(device)
            img2 = img2.float().to(device)
            e = e.to(device)
            y = y.to(device)
            risk_pred = model(img1, img2)
            valid_loss = criterion(risk_pred, y, e, model)
            valid_c = c_index(-risk_pred, y, e)
            total_val_loss += valid_loss / len(SurDataLoaders['test'])
            total_val_c += valid_c / len(SurDataLoaders['test'])
            if best_c_index < total_val_c:
                best_c_index = total_val_c
                best_epoch = epoch
                flag = 0
                if total_val_c > 0.6 and total_train_c > 0.6:
                    checkpoint = {'model': model,
                                  'model_state_dict': model.state_dict(),
                                  'optimizer': optimizer,
                                  'optimizer_state_dict': optimizer.state_dict(),
                                  'epoch': epoch
                                  }
                    torch.save(checkpoint, r"/data/gaohan/deeplearning/project/shandong/survival/result/epoch/SDS/"
                                           r"Epoch {:02d}--train {:0.2f}--test {:0.2f}--".format(epoch, total_train_c,
                                                                                                 total_val_c) + str(
                        t) + '.pkl')
            else:
                flag += 1
                if flag >= patience:
                    break

    print('\rEpoch: {}\tLoss: {:.8f}({:.8f})\tc-index: {:.8f}({:.8f})\tlr: {:g}'.format(
        epoch, total_train_loss, total_val_loss, total_train_c, total_val_c, lr), end='', flush=False)
print('-' * 100)
print('best_c_index：{},best_epoch:{}'.format(best_c_index, best_epoch))
