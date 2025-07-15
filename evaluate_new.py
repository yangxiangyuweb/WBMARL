
import matplotlib.pyplot as plt
import numpy as np
import torch
import collections
import random
from test import predict
from scipy.interpolate import interp1d
import cv2
#奖励函数随训练变化的曲线
#MIE随训练变化的曲线
#SA随着训练变化的曲线
#SCR随着训练变化的曲线


def Image_process(images,wavelegths_selected_index,bandwidths_selected):

    widths = np.array([56.25,54.01,51.93, 49.47, 47.72, 46.09, 44.94, 44.2, 43.26, 42.51, 41.7,
                    40.83, 39.89, 38.89, 37.86, 36.81, 36.1, 35.06, 34.03, 33.37, 32.41, 31.49, 30.9,30.45,30.01])  # 每个波段的带宽
    wavelengths=np.array([4940,4876,4808.036931622306, 4742.117291439937, 4677.428858401089, 4613.958667157539, 4551.69375236106,
                              4490.621148663428, 4430.7278907164155, 4372.001013171803, 4314.427550681357, 4257.994537896858,
                              4202.68900947008, 4148.498000052796, 4095.4085442967835, 4043.4076768538152, 3992.4824323756675,
                              3942.619845514114, 3893.80695092093, 3846.03078324789, 3799.2783771467703, 3753.5367672693437,
                              3708.792988267387,3662.034074999999, 3617.2470624999995])  # 每个波段的中心波长

    images=np.concatenate([images,np.expand_dims(images[-1,:,:],axis=0).repeat(2,axis=0)],axis=0)
    images = np.concatenate([ np.expand_dims(images[0, :, :], axis=0).repeat(2, axis=0),images], axis=0)

    processed_images=[]
    selected_images=[]

    for i in range(5):
        index=int(wavelegths_selected_index[i])+2
        bandwidth_selected = bandwidths_selected[i]
        wavelength_selected = wavelengths[index]
        selected_images.append(images[index,:,:])
        if bandwidth_selected<=widths[index]:
            processed_image = images[index, :, :]
        else:
            processed_image= np.zeros((256, 320))
            images_neibering = images[index-2:index+3,:,:]
            x_new=np.linspace(wavelengths[index-2], wavelengths[index+2], 50)
            x_original = wavelengths[index-2:index+3]

            images_interploted=np.zeros([50,256,320])
            for i in range(256):
                for j in range(320):
                    images_interploted[:, i, j] = np.interp(x_new, x_original, images_neibering[:, i, j])
            delta_width=(wavelengths[index+2]- wavelengths[index-2])/50
            ratio= np.abs(delta_width/widths[index])
            wavelength_start=wavelength_selected- bandwidth_selected/2
            wavelength_end = wavelength_selected + bandwidth_selected / 2
            for j in range(50):
                if x_new[j] >= wavelength_start and x_new[j] <= wavelength_end:
                    processed_image = processed_image+images_interploted[j,:,:]*ratio
        processed_images.append(processed_image)
    return np.array(processed_images),np.array(selected_images)

def image_entropy_cv(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist.astype(float) / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
    return entropy

def MIE(processed_images,background='sky'):
    IE=0
    for i in range(5):
        img = cv2.normalize(processed_images[i,:,:], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        IE=IE+image_entropy_cv(img)
    MIE=IE/processed_images.shape[0]
    return MIE

def SA(processed_images,background='sky'):
    position = {
        'sky': (110, 115, 70, 75, 149, 154, 99, 104),
        'building': (50, 55, 100, 105, 100, 105, 130, 135, 100, 105),
        'forest': (110, 115, 144, 149, 130, 135, 155, 160),
        'road': (50, 55, 100, 105, 100, 105, 130, 135)}

    Target1= processed_images[:, position[background][2]:position[background][3],position[background][0]:position[background][1]]
    Target2 = processed_images[:,position[background][6]:position[background][7], position[background][4]:position[background][5]]
    Target1=np.mean(Target1,axis=(1,2))
    Target2 = np.mean(Target2, axis=(1,2))
    Target1=(Target1-np.min(Target1))/(np.max(Target1)-np.min(Target1))
    Target2 = (Target2 - np.min(Target2)) / (np.max(Target2) - np.min(Target2))
    SA=np.arccos(np.corrcoef(Target1,Target2)[0,1])
    return SA

def SCR(processed_images,background='sky'):
    position = {
        'sky': (110, 115, 70, 75, 149, 154, 99, 104),
        'building': (50, 55, 100, 105, 100, 105, 130, 135, 100, 105),
        'forest': (110, 115, 144, 149, 130, 135, 155, 160),
        'road': (50, 55, 100, 105, 100, 105, 130, 135)}
    Target1= processed_images[:, position[background][2]:position[background][3],position[background][0]:position[background][1]]
    Target2 = processed_images[:,position[background][6]:position[background][7], position[background][4]:position[background][5]]
    Background = processed_images[:, position[background][2]-20:position[background][3]-15,position[background][0]+15:position[background][1]+20]
    Target1=np.mean(Target1,axis=(1,2))
    Target2 = np.mean(Target2, axis=(1,2))
    Background=np.std(Background,axis=(1,2))
    SCR=np.sum(np.abs((Target1-Target2)/(Background+1e-6)))/processed_images.shape[0]
    return SCR

def MADRLS():
    MIE_value_list=[]
    SA_value_list=[]
    SCR_value_list=[]

    MIE_value_nowidth_list=[]
    SA_value_nowidth_list=[]
    SCR_value_nowidth_list=[]

    time1_list=[]
    time2_list=[]

    '''1. forest 2. road 3. building 4. sky 5. plane'''
    file_idx = 'sky'
    convlstm_model_path = r'./model/static_pth/conv_lstm_multy.pth'
    if (file_idx == 'building'):
        filepath = r'D:\yangxiangyu\PythonWorks\maddpg_bandwidthSelect\MSI\building\2'
        wavelength_actor_model_dir = r'./train/pth_save/building/'
        bandwidth_actor_model_dir = r'./train/pth_save/building/'
    elif (file_idx == 'forest'):
        filepath = r'D:\yangxiangyu\PythonWorks\maddpg_bandwidthSelect\MSI\forest\1'
        wavelength_actor_model_dir = r'./train/pth_save/forest/'
        bandwidth_actor_model_dir = r'./train/pth_save/forest/'
    elif (file_idx == 'road'):
        filepath = r'D:\yangxiangyu\PythonWorks\maddpg_bandwidthSelect\MSI\road\1'
        wavelength_actor_model_dir = r'./train/pth_save/road/'
        bandwidth_actor_model_dir = r'./train/pth_save/road/'
    elif (file_idx == 'sky'):
        filepath = r'D:\yangxiangyu\PythonWorks\maddpg_bandwidthSelect\MSI\sky\1'
        wavelength_actor_model_dir = r'./train/pth_save/sky/'
        bandwidth_actor_model_dir = r'./train/pth_save/sky/'
    elif (file_idx == 'all'):
        filepath = r'D:\yangxiangyu\PythonWorks\maddpg_bandwidthSelect\MSI\building\1'
        wavelength_actor_model_dir = r'./train/pth_save/all/'
        bandwidth_actor_model_dir = r'./train/pth_save/all/'

    for j in range(100):
        wavelength_actor_model_path=wavelength_actor_model_dir+'//band'+str((99+1)*10)+'.pth'
        bandwidth_actor_model_path = bandwidth_actor_model_dir + '//width' + str((99 + 1) * 10) + '.pth'
        images,wavelengths_selected_index, wavelengths_selected, bandwidths_selected,time=predict(convlstm_model_path,
                                            wavelength_actor_model_path,bandwidth_actor_model_path,filepath)
        processed_images,selected_images=Image_process(images, wavelengths_selected_index, bandwidths_selected)

        #根据波长和带宽合成的处理图像，根据波长选择的图像
        MIE_value = MIE(processed_images,background=file_idx)
        SA_value = SA(processed_images,background=file_idx)
        SCR_value = SCR(processed_images,background=file_idx)
        MIE_value_nowidth=MIE(selected_images,background=file_idx)
        SA_value_nowidth = SA(selected_images, background=file_idx)
        SCR_value_nowidth = SCR(selected_images, background=file_idx)

        MIE_value_list.append(MIE_value)
        SA_value_list.append(SA_value)
        SCR_value_list.append(SCR_value)
        MIE_value_nowidth_list.append(MIE_value_nowidth)
        SA_value_nowidth_list.append(SA_value_nowidth)
        SCR_value_nowidth_list.append(SCR_value_nowidth)
        time1_list.append(time[0])
        time2_list.append(time[1])
        print(1+j,wavelengths_selected_index,f'MIE:{MIE_value}, SA:{SA_value}, SCR:{SCR_value}, time1:{time[0]},time2:{time[1]}')
        print(1 + j, wavelengths_selected_index, f'MIE_nowidth:{MIE_value_nowidth}, SA_nowidth:{SA_value_nowidth}, SCR_nowidth:{SCR_value_nowidth}')

import os
def read_hyper1(filepath):
    imglist=[]
    qianzhui='building'
    for i in range(len(os.listdir(filepath))):
        imgname=qianzhui+str(i)+'.tif'
        imgpath = os.path.join(filepath, imgname)
        hyper = cv2.imread(imgpath, -1)
        imglist.append(hyper)
    imglist=np.array(imglist)
    return imglist

def Eva():
    '''1. forest 2. road 3. building 4. sky 5. plane'''
    filepath = r'./data/1'
    data = read_hyper1(filepath).astype(np.float64)
    selected_images = data[np.array([14, 12, 11,6, 4]),:,:]
    # 根据波长和带宽合成的处理图像，根据波长选择的图像
    file_idx = 'building'
    MIE_value = MIE(selected_images, background=file_idx)
    SA_value = SA(selected_images, background=file_idx)
    SCR_value = SCR(selected_images, background=file_idx)
    print(file_idx,f'MIE:{MIE_value}, SA:{SA_value}, SCR:{SCR_value}')
Eva()


