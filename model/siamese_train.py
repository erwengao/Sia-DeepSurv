from torch.utils.data.sampler import WeightedRandomSampler, SequentialSampler, SubsetRandomSampler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, accuracy_score
import csv
from project.shandong.survival.model.h5dataset import MyFolder
import torchio.transforms as transforms
import torchio as tio
from project.shandong.survival.model.siamese_net import generate_resmodel
import pandas as pd
import datetime

t = datetime.date.today()
# 检查GPU是否可用
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
device_ids = [2]
# 数据增强
data_transforms = {
    'train': transforms.Compose([
        tio.Resample(1),
        transforms.ToCanonical(p=0.5, copy=False),
        transforms.RandomFlip(flip_probability=0.5),
        transforms.RandomNoise(),
        transforms.ZNormalization()
    ]),
    'test': transforms.Compose([
        tio.Resample(1),
        transforms.ZNormalization()
    ]),
}
data_dir = r'/data/gaohan/deeplearning/project/shandong/survival/train_test/3d_split'
image_datasets = {x: MyFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'test']}
# 采样
# num_samples = 700
sameler_weight = [1 if image_datasets['train'].imgs[i][1] == 1 else 1 for i in range(len(image_datasets['train'].imgs))]
sampler = {
    # 'train': WeightedRandomSampler(sameler_weight, num_samples=num_samples, replacement=True),
    'train': SubsetRandomSampler(
        np.random.choice(range(len(image_datasets['train'])), len(image_datasets['train']), replace=False)),
    'test': SubsetRandomSampler(
        np.random.choice(range(len(image_datasets['test'])), len(image_datasets['test']), replace=False))
}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64 * len(device_ids), sampler=sampler[x])
               for x in ['train', 'test']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,shuffle=True)
#                  for x in ['train', 'test']}
dataset_sizes = {x: len(dataloaders[x].sampler) for x in ['train', 'test']}

# 训练模型
# model = siamese()
model = generate_resmodel(10, pretrain=True)
# criterion = BCEFocalLoss()
# criterion = nn.MSELoss()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=0)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=5, verbose=True, cooldown=2)
num_epochs = 30
# model = model.to(device)


if len(device_ids) > 1:
    print("Let's use", len(device_ids), "GPUs!")
    model = nn.DataParallel(model, device_ids=device_ids)
model.to(device)

trainacc_a = []
trainacc_b = []
testacc_a = []
testacc_b = []
trainrecall_a = []
trainrecall_b = []
testrecall_a = []
testrecall_b = []
trainloss = []
testloss = []
result_path = r'/data/gaohan/deeplearning/project/shandong/survival/result/csv/feature/sia/sia'+str(t)+'.csv'

with open(result_path, 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    # 构建表头
    csv_writer.writerow(["epoch", "Train Accuracy-a", "Train Accuracy-b",
                         "Test Accuracy-a", "Test Accuracy-b",
                         "Train Recall-a", "Train Recall-b",
                         "Test Recall-a", "Test Recall-b",
                         "Train Loss", "Test Loss"])
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 80)

        for phase in ['train', 'test']:
            # Iterate over data.
            feature_train = {}
            if phase == 'train':
                model.train()  # Set model to training mode
                train_loss = 0.0
                train_acc = 0.0
                train_true_labels = []
                train_pred_a = []
                train_pred_b = []

                for input_a, input_b, labels, ids in dataloaders['train']:
                    input_a = input_a.float().to(device)
                    input_b = input_b.float().to(device)
                    labels = labels.to(device)
                    # ids = ids.to(device)
                    train_true_labels.append(labels)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    output_a, output_b, output = model(input_a, input_b)
                    # print(output)

                    # 训练集数据特征
                    for i in range(len(ids)):
                        feature_train[ids[i]] = output[i].detach().cpu().numpy()  ##训练集output有梯度 所以要detach

                    pred_a = torch.ge(output_a, 0.5)
                    pred_b = torch.ge(output_b, 0.5)
                    train_pred_a.append(torch.squeeze(pred_a, -1))
                    train_pred_b.append(torch.squeeze(pred_b, -1))
                    ## 反传和更新
                    loss_a = criterion(output_a.view(-1), labels.float().view(-1))
                    loss_b = criterion(output_b.view(-1), labels.float().view(-1))
                    loss = 0.5 * loss_a + 0.5 * loss_b
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * input_a.size(0)

                train_epoch_loss = train_loss / dataset_sizes['train']
                train_true_labels = [i.item() for item in train_true_labels for i in item]
                train_pred_a = [i.item() for item in train_pred_a for i in item]
                train_pred_b = [i.item() for item in train_pred_b for i in item]
                train_epoch_acc_a = accuracy_score(torch.tensor(train_true_labels), torch.tensor(train_pred_a))
                train_epoch_acc_b = accuracy_score(torch.tensor(train_true_labels), torch.tensor(train_pred_b))
                train_epoch_recall_a = recall_score(torch.tensor(train_true_labels), torch.tensor(train_pred_a))
                train_epoch_recall_b = recall_score(torch.tensor(train_true_labels), torch.tensor(train_pred_b))

                trainloss.append(train_epoch_loss)
                trainacc_a.append(train_epoch_acc_a)
                trainacc_b.append(train_epoch_acc_b)
                trainrecall_a.append(train_epoch_recall_a)
                trainrecall_b.append(train_epoch_recall_b)
                a = sum(train_true_labels)
                print(a, dataset_sizes['train'])
                deep_feature_train = pd.DataFrame.from_dict(feature_train, orient='index')
                deep_feature_train.to_csv(
                    '/data/gaohan/deeplearning/project/shandong/survival/result/csv/feature/train/{}_'.format(epoch)
                    + str(t) + '.csv')

            feature_test = {}
            if phase == 'test':
                with torch.no_grad():
                    model.eval()  # Set model to evaluate mode
                    test_loss = 0.0
                    test_acc = 0.0
                    test_true_labels = []
                    test_pred_a = []
                    test_pred_b = []

                    for input_a, input_b, labels, ids in dataloaders['test']:
                        input_a = input_a.float().to(device)
                        input_b = input_b.float().to(device)
                        labels = labels.to(device)
                        # ids = ids.to(device)
                        test_true_labels.append(labels)
                        # zero the parameter gradients
                        optimizer.zero_grad()
                        # forward
                        output_a, output_b, output = model(input_a, input_b)

                        # 测试集数据特征
                        for i in range(len(ids)):
                            feature_test[ids[i]] = output[i].cpu().numpy()

                        pred_a = torch.ge(output_a, 0.5)
                        pred_b = torch.ge(output_b, 0.5)
                        test_pred_a.append(torch.squeeze(pred_a, -1))
                        test_pred_b.append(torch.squeeze(pred_b, -1))
                        # BCELOSS
                        loss_a = criterion(output_a.view(-1), labels.float().view(-1))
                        loss_b = criterion(output_b.view(-1), labels.float().view(-1))
                        loss = 0.5 * loss_a + 0.5 * loss_b
                        test_loss += loss.item() * input_a.size(0)

                    test_epoch_loss = test_loss / dataset_sizes['test']
                    test_true_labels = [i.item() for item in test_true_labels for i in item]
                    test_pred_a = [i.item() for item in test_pred_a for i in item]
                    test_pred_b = [i.item() for item in test_pred_b for i in item]
                    test_epoch_acc_a = accuracy_score(torch.tensor(test_true_labels), torch.tensor(test_pred_a))
                    test_epoch_acc_b = accuracy_score(torch.tensor(test_true_labels), torch.tensor(test_pred_b))
                    test_epoch_recall_a = recall_score(torch.tensor(test_true_labels), torch.tensor(test_pred_a))
                    test_epoch_recall_b = recall_score(torch.tensor(test_true_labels), torch.tensor(test_pred_b))

                    testloss.append(test_epoch_loss)
                    testacc_a.append(test_epoch_acc_a)
                    testacc_b.append(test_epoch_acc_b)
                    testrecall_a.append(test_epoch_recall_a)
                    testrecall_b.append(test_epoch_recall_b)
                    a = sum(test_true_labels)
                    print(a, dataset_sizes['test'])
                    deep_feature_test = pd.DataFrame.from_dict(feature_test, orient='index')
                    deep_feature_test.to_csv(
                        '/data/gaohan/deeplearning/project/shandong/survival/result/csv/feature/test/{}_'.format(epoch)
                        + str(t) + '.csv')

        # 保存特征
        # deep_feature = pd.DataFrame.from_dict(feature, orient='index')
        # deep_feature.to_csv('/data/gaohan/deeplearning/project/shandong/survival/result/csv/feature/{}.csv'.format(epoch))

        # 保存模型
        checkpoint = {'model': model,
                      'model_state_dict': model.state_dict(),
                      'optimizer': optimizer,
                      'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': epoch
                      }
        torch.save(checkpoint, r"/data/gaohan/deeplearning/project/shandong/survival/result/epoch/sia/"
                               r"Epoch {:02d}".format(epoch)+str(t)+'.pkl')
        print("epoch :{}, \n"
              "Train Accuracy-a :{}, Train Accuracy-b :{},\n"
              "Test Accuracy-a :{}, Test Accuracy-b :{}, \n"
              "Train Recall-a :{}, Train Recall-b :{}, \n"
              "Test Recall-a :{}, Test Recall-b :{}\n, "
              "Train Loss :{}, Test Loss :{}".format(epoch, trainacc_a[epoch], trainacc_b[epoch],
                                                     testacc_a[epoch], testacc_b[epoch],
                                                     trainrecall_a[epoch], trainrecall_b[epoch],
                                                     testrecall_a[epoch], testrecall_b[epoch],
                                                     trainloss[epoch], testloss[epoch]))
        csv_writer.writerow(["{:02d}".format(epoch),
                             "{:.5f}".format(trainacc_a[epoch]), "{:.5f}".format(trainacc_b[epoch]),
                             "{:.5f}".format(testacc_a[epoch]), "{:.5f}".format(testacc_b[epoch]),
                             "{:.5f}".format(trainrecall_a[epoch]), "{:.5f}".format(trainrecall_b[epoch]),
                             "{:.5f}".format(testrecall_a[epoch]), "{:.5f}".format(testrecall_b[epoch]),
                             "{:.5f}".format(trainloss[epoch]), "{:.5f}".format(testloss[epoch])])
plt.subplot(3, 1, 1)
loss_values = trainloss
val_loss_values = testloss
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'r', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(3, 1, 2)
acc_values = trainacc_a
val_acc_values = testacc_a
plt.plot(epochs, acc_values, 'r', label='Training acc -a')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc -a')
plt.title('Training and validation accuracy - a')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.subplot(3, 1, 3)
acc_values = trainacc_b
val_acc_values = testacc_b
plt.plot(epochs, acc_values, 'r', label='Training acc-b')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc-b')
plt.title('Training and validation accuracy-b')
plt.xlabel('Epochs')
plt.ylabel('Acc')
plt.legend()
plt.savefig(r'/data/gaohan/deeplearning/project/shandong/survival/result/figure/sia'+str(t)+'.png')
plt.show()
