import h5py
import os
import csv
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = checkpoint['model']  # 提取网络结构
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载网络权重参数
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器参数

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


##病人级别预测分类结果
def patient_acc(h5_path, model_path, csv_path, save_path_train, save_path_test):
    csv_label = pd.read_csv(csv_path)
    csv_label['ID'] = csv_label['ID'].astype(str)
    train_ID = csv_label.loc[csv_label['TRAIN'] == 1].ID.tolist()
    test_ID = csv_label.loc[csv_label['TRAIN'] == 0].ID.tolist()

    with open(save_path_train, 'w', newline='') as f: ##创建csv
        writer = csv.writer(f)          ##创建写的对象
        writer.writerow(["ID", "LABEL", "before", "after"])      ##写入列的名称
        model = load_checkpoint(model_path) ##pytorch
        train_acc_before = 0
        train_acc_after = 0
        for i in os.listdir(h5_path):  ## i 为ID
            if str(int(i)) in train_ID:
                print(i)
                img_before = None
                img_after = None
                for j in os.listdir(os.path.join(h5_path, i)): ##j为每个病人治疗前后的ct
                    if j == '1.h5':
                        data_before = h5py.File(os.path.join(h5_path, i, j), 'r')
                        img_before = np.array(data_before['image'][:])
                        img_before = img_before[None, None, :, :, :]
                        img_before = torch.tensor(img_before, dtype=torch.float32)
                        img_before = np.concatenate((img_before, img_before, img_before), axis=1)
                        img_before = torch.from_numpy(img_before)
                    elif j == '2.h5':
                        data_after = h5py.File(os.path.join(h5_path, i, j), 'r')
                        img_after = np.array(data_after['image'][:])
                        img_after = img_after[None, None, :, :, :]
                        img_after = torch.tensor(img_after, dtype=torch.float32)
                        img_after = np.concatenate((img_after, img_after, img_after), axis=1)
                        img_after = torch.from_numpy(img_after)
                output_before, output_after, feature = model(img_before, img_after)
                before = 0 if output_before.item() < 0.5 else 1
                if before == csv_label.loc[csv_label["ID"] == str(int(i))]['LABEL'].values:
                    train_acc_before = train_acc_before + 1
                after = 0 if output_after.item() < 0.5 else 1
                if after == csv_label.loc[csv_label["ID"] == str(int(i))]['LABEL'].values:
                    train_acc_after = train_acc_after + 1
                writer.writerow([i, csv_label.loc[csv_label["ID"] == str(int(i))]['LABEL'].values, output_before.item(), output_after.item()])

        ##病人级别acc
        trainaccbefore = train_acc_before/len(train_ID)
        trainaccafter = train_acc_after/len(train_ID)
        print("train_acc_before: {},train_acc_after: {}".format(trainaccbefore, trainaccafter))

    with open(save_path_test, 'w', newline='') as f:  ##创建csv
        writer = csv.writer(f)  ##创建写的对象
        writer.writerow(["ID", "LABEL", "before", "after"])  ##写入列的名称
        model = load_checkpoint(model_path)  ##pytorch
        test_acc_before = 0
        test_acc_after = 0
        for i in os.listdir(h5_path):  ## i 为ID
            if str(int(i)) in test_ID:
                print(i)
                img_before = None
                img_after = None
                for j in os.listdir(os.path.join(h5_path, i)):  ##j为每个病人治疗前后的ct
                    if j == '1.h5':
                        data_before = h5py.File(os.path.join(h5_path, i, j), 'r')
                        img_before = np.array(data_before['image'][:])
                        img_before = img_before[None, None, :, :, :]
                        img_before = torch.tensor(img_before, dtype=torch.float32)
                        img_before = np.concatenate((img_before, img_before, img_before), axis=1)
                        img_before = torch.from_numpy(img_before)
                    elif j == '2.h5':
                        data_after = h5py.File(os.path.join(h5_path, i, j), 'r')
                        img_after = np.array(data_after['image'][:])
                        img_after = img_after[None, None, :, :, :]
                        img_after = torch.tensor(img_after, dtype=torch.float32)
                        img_after = np.concatenate((img_after, img_after, img_after), axis=1)
                        img_after = torch.from_numpy(img_after)
                output_before, output_after, feature = model(img_before, img_after)
                before = 0 if output_before.item() < 0.5 else 1
                if before == csv_label.loc[csv_label["ID"] == str(int(i))]['LABEL'].values:
                    test_acc_before = test_acc_before + 1
                after = 0 if output_after.item() < 0.5 else 1
                if after == csv_label.loc[csv_label["ID"] == str(int(i))]['LABEL'].values:
                    test_acc_after = test_acc_after + 1
                writer.writerow([i, csv_label.loc[csv_label["ID"] == str(int(i))]['LABEL'].values, output_before.item(), output_after.item()])

        ##病人级别acc
        testaccbefore = test_acc_before / len(train_ID)
        testaccafter = test_acc_after / len(train_ID)
        print("test_acc_before: {},test_acc_after: {}".format(testaccbefore, testaccafter))
        return


patient_acc(r"/data/gaohan/deeplearning/project/shandong/survival/h5/3d",
    r"/data/gaohan/deeplearning/project/shandong/survival/result/epoch/sia/"
    r"Epoch 03-Train Loss - Test Loss0.68339.pkl",
    r"/data/gaohan/deeplearning/project/shandong/survival/information/label.csv",
    r"/data/gaohan/deeplearning/project/shandong/survival/result/patient/train.csv",
    r"/data/gaohan/deeplearning/project/shandong/survival/result/patient/test.csv",)






## 病人级别auc
def patient_auc(train_path,test_path,save_path):
    data_train = pd.read_csv(train_path)
    data_train = np.array(data_train)
    data_test = pd.read_csv(test_path)
    data_test = np.array(data_test)
    fpr0, tpr0, _ = roc_curve(data_train[:,1],data_train[:,2])
    fpr1, tpr1, _ = roc_curve(data_test[:,1],data_test[:,2])
    train_auc = auc(fpr0, tpr0)
    test_auc = auc(fpr1, tpr1)
    print("train auc%s" % train_auc)
    print("test auc%s" % test_auc)
    plt.figure(figsize=(5, 5))
    plt.title('\n  train auc:' + str(round(train_auc, 3)) + 'test auc' + str(round(test_auc, 3)))
    plt.xlabel("fpr")
    plt.ylabel("tpr")
    plt.plot(fpr0, tpr0, color='red', label='train')
    plt.plot(fpr1, tpr1, color='blue', label='test')
    plt.savefig(save_path)
    plt.show()
    return

patient_auc(r"/data/gaohan/deeplearning/project/shandong/survival/result/patient/train.csv",
            r"/data/gaohan/deeplearning/project/shandong/survival/result/patient/test.csv",
            r"/data/gaohan/deeplearning/project/shandong/survival/result/figure/sia.csv")






















