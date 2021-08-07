import SimpleITK as sitk
import h5py
import os
import cv2
import pandas as pd
import numpy as np
import PIL.Image

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def window_transform(img, windowWidth, windowCenter, normal=False):
    minWindow = float(windowCenter) - 0.5 * float(windowWidth)
    newimg = (img - minWindow) / float(windowWidth)
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    if not normal:
        newimg = (newimg * 255).astype('uint8')
    return newimg


# 调整CT图像的窗宽窗位
def setDicomWinWidthWinCenter(img_data, winwidth, wincenter, rows, cols):
    img_temp = img_data
    img_temp.flags.writeable = True
    min = (2 * wincenter - winwidth) / 2.0 + 0.5
    max = (2 * wincenter + winwidth) / 2.0 + 0.5
    dFactor = 255.0 / (max - min)

    for i in np.arange(rows):
        for j in np.arange(cols):
            img_temp[i, j] = int((img_temp[i, j] - min) * dFactor)

    min_index = img_temp < 0
    img_temp[min_index] = 0
    max_index = img_temp > 255
    img_temp[max_index] = 255

    return img_temp


# 3d_h5(生存)
def readash5(savepath, totalpath, label_csvpath):
    # 读取勾画的层，存为h5
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # 找出对应ID的label
    totalpath = totalpath
    csv = pd.read_csv(label_csvpath)
    csv["ID"] = csv["ID"].astype(str)
    csv = csv.set_index(["ID"])
    label = csv["LABEL"]
    time = csv["TIME"]
    # 选取勾画了的层存为.h5
    for i in os.listdir(totalpath):  # i为id号
        print(i)
        label_i = label[str(int(i))]
        time_i = time[str(int(i))]
        maskpath = []
        oripath = []
        patientpath = totalpath + os.sep + i
        flag = None
        for stage in os.listdir(patientpath):  # 每个病人有治疗前后两个图像
            if os.path.isdir(patientpath+os.sep+stage):
                flag = 1
                for j in os.listdir(patientpath + os.sep + stage):
                    filename, fileextension = os.path.splitext(j)
                    if not fileextension == '.nii':
                        continue
                    if 'label' in filename:
                        maskpath = patientpath + os.sep + stage + os.sep + j
                    else:
                        oripath = patientpath + os.sep + stage + os.sep + j
                save(maskpath, oripath, i, label_i, time_i, stage)
            else:
                if 'label' in stage:
                    maskpath = patientpath + os.sep + stage
                else:
                    oripath = patientpath + os.sep + stage
        if not flag:
            save(maskpath, oripath, i, label_i, time_i)

    return


def save(maskpath, oripath, i, label_i, time_i, stage=None):
    file0 = sitk.ReadImage(maskpath)
    mask = sitk.GetArrayFromImage(file0)  ##此时mask为所有层
    layersori = []
    for k in range(mask.shape[0]):
        if mask[k, :, :].sum() > 0:  ##没有勾画的地方sum为0 ，没勾画的地方即为没有肿瘤
            layersori.append(k)  ##layer中存放的为勾画的层
    layers = [i for i in range(layersori[len(layersori) // 2] - 5, layersori[len(layersori) // 2] + 5)]  ##取最中间10层
    file1 = sitk.ReadImage(oripath)
    ori = sitk.GetArrayFromImage(file1)
    p = savepath + os.sep + i
    if not os.path.exists(p):
        os.makedirs(p)
    pictures = []
    XMIN = []
    XMAX = []
    YMIN = []
    YMAX = []
    for l in layers:  # 遍历勾画的层
        if l not in layersori:
            pictures.append(np.zeros((512, 512)))
        else:
            img_ori = ori[l]
            mask_ori = mask[l]
            tumer_wide = 400
            tumer_center = 40
            r = img_ori.shape[0]
            c = img_ori.shape[1]
            # picture = window_transform(img_ori, tumer_wide, tumer_center, normal=False)  # 归一化到0-1
            picture = setDicomWinWidthWinCenter(img_ori, tumer_wide, tumer_center, r, c)  # 归一化到0-1
            pictures.append(picture)
            for x in range(img_ori.shape[0]):  # 选出只有mask的部分
                for y in range(img_ori.shape[1]):
                    if mask_ori[x, y] == 0:
                        img_ori[x, y] = 0
            location = np.nonzero(img_ori)  # 返回非零元素的索引 二维图片第一个array储存行，第二个array储存列
            xmin = int(location[0][0])
            xmax = int(location[0][-1])
            location[1].sort()  # 二维是按行的顺序排列的，所以行不用sort
            ymin = int(location[1][0])
            ymax = int(location[1][-1])
            XMIN.append(xmin)
            XMAX.append(xmax)
            YMIN.append(ymin)
            YMAX.append(ymax)

    Xmin = min(XMIN)
    Xmax = max(XMAX)
    Ymin = min(YMIN)
    Ymax = max(YMAX)
    pictures_cube = []
    for picture in pictures:
        picture = picture[Xmin:Xmax, Ymin:Ymax]
        picture = cv2.resize(picture.astype('float64'), (64, 64))  # 选取只有肿瘤的部位
        pictures_cube.append(picture)

    pictures64 = np.array(pictures_cube)
    # 病人级别归一化
    mean = pictures64.mean()
    std = pictures64.std()
    out = (pictures64 - mean) / std
    if stage:
        with h5py.File(p + os.sep + stage + '.h5', 'w') as hf:
            print(p + os.sep + stage + '.h5')
            hf.create_dataset("image", data=out)  # 每个病人图像
            hf.create_dataset("label", data=[label_i])  # 是否发生事件（LRFS）
            hf.create_dataset("time", data=[time_i])  # 每个病人发生事件时间 （不做生存不需要这步）

    else:
        for stage in range(1, 3):
            with h5py.File(p + os.sep + str(stage) + '.h5', 'w') as hf:
                print(p + os.sep + str(stage) + '.h5')
                hf.create_dataset("image", data=out)  # 每个病人图像
                hf.create_dataset("label", data=[label_i])  # 是否发生事件（LRFS）                hf.create_dataset("time", data=[time_i])      #每个病人发生事件时间 （不做生存不需要这步）
                hf.create_dataset("time", data=[time_i])      #每个病人发生事件时间 （不做生存不需要这步）

    return


if __name__ == '__main__':
    savepath = r"/data/gaohan/deeplearning/project/shandong/survival/h5/3d_survival_new"
    totalpath = r"/data/gaohan/deeplearning/project/shandong/survival/data_mod"
    label_csvpath = r"/data/gaohan/deeplearning/project/shandong/survival/information/label.csv"
    readash5(savepath, totalpath, label_csvpath)
