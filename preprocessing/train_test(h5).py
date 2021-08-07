import numpy as np
import pandas as pd
import h5py
import os
import shutil
import warnings


##3D 将H5文件按选练和测试划分，保存在savepath下
def label_classification(h5path,csvpath,savepath):           ##csvpath 为训练和测试集csv的根目录
    ##建立文件夹
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    train_path = savepath + os.sep + 'train'
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = savepath + os.sep + 'test'
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    ##划分训练测试,csv中TRAIN为1 表示为训练
    csv = pd.read_csv(csvpath)
    csv['ID'] = csv['ID'].astype(str)
    train_ID = csv.loc[csv['TRAIN1'] == 1].ID.tolist()
    test_ID = csv.loc[csv['TRAIN1'] == 0].ID.tolist()


    #将不同病人的h5文件根据CSV拷贝到train 和 test 两个文件夹下
    for patientdir in os.listdir(h5path):
        oripath = h5path + os.sep + patientdir
        trainpath = train_path + os.sep + patientdir
        testpath = test_path + os.sep + patientdir
        if str(int(patientdir)) in train_ID:
            shutil.copytree(oripath, trainpath)
        elif str(int(patientdir)) in test_ID:
            shutil.copytree(oripath, testpath)
        else:
             warnings.warn('The patient neither in train set nor in test set')
    return


if __name__ == '__main__':
    h5path = r"/data/gaohan/deeplearning/project/shandong/survival/h5/3d_survival_new"
    csvpath = r"/data/gaohan/deeplearning/project/shandong/survival/information/label.csv"
    savepath = r"/data/gaohan/deeplearning/project/shandong/survival/train_test/3d_split_1"
    label_classification(h5path, csvpath, savepath)




