import os
import torch
import numpy as np
import pandas as pd
import h5py

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


def my_loader(path):
    data = h5py.File(path, 'r')
    img = np.array(data['image'][:])
    img = img[None, None, :, :, :]
    img = np.concatenate((img, img, img), axis=1)  # 模型验证时，没有batch（一个病人一个病人输入），第一个位置增加一个维度
    img = torch.tensor(img, dtype=torch.float32)

    return img


def risk_score(Patient_dir, Model_path, Save_path):
    model = load_checkpoint(Model_path)
    risk = {}
    for i in os.listdir(Patient_dir):  ## i 为ID
        path1 = None
        path2 = None
        for j in os.listdir(os.path.join(Patient_dir, i)):  ##j为1.h5/2.h5
            if os.path.splitext(j)[-2] == '1':
                path1 = os.path.join(Patient_dir, i, j)
            if os.path.splitext(j)[-2] == '2':
                path2 = os.path.join(Patient_dir, i, j)
        img1 = my_loader(path1)
        img2 = my_loader(path2)
        pred_risk = model(img1, img2)
        risk[i] = pred_risk.view(-1).detach().cpu().numpy()
    deep_feature_train = pd.DataFrame.from_dict(risk, orient='index')
    deep_feature_train.to_csv(Save_path)


patient_dir = r"/data/gaohan/deeplearning/project/shandong/survival/h5/3d_survival_new"
model_path = r"/data/gaohan/deeplearning/project/shandong/survival/result/epoch/SDS/Epoch 05-2021-07-01.pkl"
save_path = r"/data/gaohan/deeplearning/project/shandong/survival/result/csv/risk/Epoch 05-2021-07-01.csv"
risk_score(patient_dir, model_path, save_path)
