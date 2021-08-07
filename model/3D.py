from torch.utils.data.sampler import WeightedRandomSampler, SequentialSampler,SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score,accuracy_score
import csv
from common.myresnet import generate_model
from project.shandong.survival.model.h5dataset import MyFolder
import torchio.transforms as transforms

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
##检查GPU是否可用
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

#数据增强
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomFlip(flip_probability=1),
        transforms.RandomBlur(std=0.3, p=1),
        transforms.RandomAffine(p=1),
        transforms.ZNormalization()
    ]),
    'test': transforms.Compose([
        transforms.ZNormalization()
    ]),
}

data_dir = r'/data/gaohan/deeplearning/project/shandong/survival/train_test/3d_split1'
image_datasets = {x: MyFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'test']}
# image_datasets = MyFolder(os.path.join(data_dir))
#采样
num_samples = 700
sameler_weight = [1 if image_datasets['train'].imgs[i][1] == 1 else 1 for i in range(len(image_datasets['train'].imgs))]
sampler = {
    'train': WeightedRandomSampler(sameler_weight, num_samples=num_samples, replacement=True),
    'test': SubsetRandomSampler(np.random.choice(range(len(image_datasets['test'])),len(image_datasets['test']),replace=False))
}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, sampler=sampler[x])
                 for x in ['train', 'test']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,shuffle=True)
#                  for x in ['train', 'test']}
dataset_sizes = {x: len(dataloaders[x].sampler) for x in ['train', 'test']}

# 训练模型
model_base = generate_model(10)
# model_base.apply(weights_init_normal)
for param in model_base.parameters():
    param.requires_grad = False
channels_in = model_base.fc.in_features
# model_base.fc = nn.Linear(channels_in, 1)
model_base.fc = nn.Sequential(nn.Linear(channels_in, 500), nn.Dropout(0.5),nn.Linear(500,20),nn.Linear(20,1),nn.Sigmoid())
# criterion = BCEFocalLoss()
# criterion = nn.MSELoss()
criterion = nn.BCELoss()
optimizer = optim.SGD(model_base.fc.parameters(), lr=1e-3, momentum=0.9, weight_decay=0.0001)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2, verbose=True, cooldown=2)
model = model_base
num_epochs = 50
model = model.to(device)

trainacc = []
testacc = []
recall_train = []
trainloss = []
testloss = []
recall_test = []
result_path = r'/data/gaohan/deeplearning/project/shandong/survival/result/csv/3d.csv'

with open(result_path, 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    ##构建表头
    csv_writer.writerow(["epoch", "Train Accuracy", "Test Accuracy", "TrainLoss", "TestLoss", "trainrecall", "testrecall","trainacc","testacc"])
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 100)
        for phase in ['train', 'test']:
            # Iterate over data.
            if phase == 'train':
                model.train()  # Set model to training mode
                train_loss = 0.0
                train_acc = 0.0
                train_true_labels = []
                train_pred_labels = []
                for inputs, labels in dataloaders['train']:

                    # bs, ncrops, c, h, w = inputs.size()
                    inputs = inputs.float().to(device)
                    labels = labels.float().to(device)
                    train_true_labels.append(labels.view(-1))

                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    outputs = model(inputs)
                    # outputs_crop = model(inputs.view(-1, c, h, w))
                    # outputs = outputs_crop.view(bs,ncrops,-1).mean(1)
                    preds = torch.ge(outputs, 0.5)
                    # print(sum(preds),len(preds.numpy()))
                    train_pred_labels.append(torch.squeeze(preds, -1))
                    # loss_weight = [3 if labels[i] == 1 else 1 for i in range(len(labels.numpy()))]
                    # loss_weight = torch.tensor(loss_weight).float()
                    loss_weight = labels * 0 + 1
                    balanced_weight = labels * 8 + 1
                    ## BCELOSS
                    loss = criterion(outputs.view(-1), labels.view(-1))
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * inputs.size(0)


                train_epoch_loss = train_loss / dataset_sizes['train']
                train_true_labels = [i.item() for item in train_true_labels for i in item]
                train_pred_labels = [i.item() for item in train_pred_labels for i in item]
                train_epoch_acc = accuracy_score(torch.tensor(train_true_labels), torch.tensor(train_pred_labels))
                recall_train_epoch = recall_score(torch.tensor(train_true_labels), torch.tensor(train_pred_labels))

                trainloss.append(train_epoch_loss)
                trainacc.append(train_epoch_acc)
                recall_train.append(recall_train_epoch)
                a = sum(train_true_labels)
                print(a, dataset_sizes['train'])
            if phase == 'test':
                model.eval()  # Set model to evaluate mode
                test_loss = 0.0
                test_acc = 0.0
                test_true_labels = []
                test_pred_labels = []

                for inputs, labels in dataloaders['test']:
                    inputs = inputs.float().to(device)
                    labels = labels.float().to(device)
                    test_true_labels.append(labels)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    outputs = model(inputs)
                    preds = torch.ge(outputs, 0.5)
                    test_pred_labels.append(torch.squeeze(preds, -1))
                    balanced_weight = labels * 8 + 1
                    loss = criterion(outputs.view(-1), labels)
                    test_loss += loss.item() * inputs.size(0)

                test_epoch_loss = test_loss / dataset_sizes['test']
                test_true_labels = [i.item() for item in test_true_labels for i in item]
                test_pred_labels = [i.item() for item in test_pred_labels for i in item]
                test_epoch_acc = accuracy_score(torch.tensor(test_true_labels), torch.tensor(test_pred_labels))
                recall_test_epoch = recall_score(torch.tensor(test_true_labels), torch.tensor(test_pred_labels))

                testloss.append(test_epoch_loss)
                testacc.append(test_epoch_acc)
                recall_test.append(recall_test_epoch)
                a = sum(test_true_labels)
                print(a, dataset_sizes['test'])
                scheduler.step(test_epoch_loss)
        # 保存模型
        checkpoint = {'model': model,
                      'model_state_dict': model.state_dict(),
                      'optimizer': optimizer,
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch}
        torch.save(checkpoint, r"/data/gaohan/deeplearning/project/shandong/survival/result/epoch/3D/Epoch {:02d}-Train Accuracy{:.5f}-TrainLoss{:.5f}-Test Accuracy-{:.5f}-TestLoss{:.5f}.pkl".format(epoch, trainacc[epoch],trainloss[epoch], testacc[epoch],testloss[epoch]))
        print("Epoch {}, \n TrainLoss: {},TestLoss: {},\n trainrecall:{},testrecall:{},\n trainacc:{},testacc:{}".format(epoch,trainloss[epoch], testloss[epoch],recall_train[epoch], recall_test[epoch],trainacc[epoch], testacc[epoch]))
        csv_writer.writerow(["{:02d}".format(epoch),"{:.5f}".format(trainloss[epoch]),"{:.5f}".format(testloss[epoch]),"{:.5f}".format(recall_train[epoch]),"{:.5f}".format(recall_test[epoch]),"{:.5f}".format(trainacc[epoch]),"{:.5f}".format(testacc[epoch])])

plt.subplot(1,2,1)
loss_values = trainloss
val_loss_values = testloss
epochs = range(1,len(loss_values)+1)
plt.plot(epochs,loss_values,'r', label='Training loss')
plt.plot(epochs,val_loss_values,'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1,2,2)
acc_values = trainacc
val_acc_values = testacc
plt.plot(epochs,acc_values,'r',label='Training acc')
plt.plot(epochs,val_acc_values,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.savefig(r'/data/gaohan/deeplearning/project/shandong/survival/result/figure/3d.png')
plt.show()