import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random
import albumentations as A

import cv2
import PIL.Image as Image
from tqdm import tqdm

from multiprocessing import Pool
from functools import partial
import copy


def rotate(src,angle):
    height, width, channel = src.shape
    matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    dst = cv2.warpAffine(src, matrix, (width, height))
    return dst

# albumentation module
def get_transformer():
    
    transform = A.Compose([
        # A.Equalize         (mode='cv', by_channels=False, mask=None, mask_params=(), always_apply=False, p=0.15),
        A.ColorJitter      (brightness=0.2,contrast=0.2,saturation=0.02,hue=0.02,p=0.35),
        # A.RandomBrightness (limit=0.15, always_apply=False, p=0.35),
        # A.RandomContrast   (limit=0.15, always_apply=False, p=0.35),
        A.Flip             (p=0.75),
        A.CLAHE            (always_apply=False, p=0.75, clip_limit=(1, 10), tile_grid_size=(8, 8))
    ])
    
    return transform


# transform module
def transform_img(transformer,i_dst_path,i_pathidx):
    '''
        # locked
            transformer
            path_list
            i_dst_path
        # iterable
            idx
    '''
    
    # i_path  = random.sample(path_list,1)[0]
    i_path = i_pathidx[0]
    idx    = i_pathidx[1]
    
    i_fname = f'{idx:06d}.jpg'
    i_dstfpath  = os.path.join(i_dst_path,i_fname)
    if os.path.isfile(i_dstfpath):
        return None
    # i_img   = Image.open(i_path)
    # i_img   = np.asarray(i_img,dtype=np.uint8)
    try:
        i_img = cv2.imread(i_path)
        i_img = cv2.cvtColor(i_img,cv2.COLOR_BGR2RGB)
    except:
        f = open('./__aug-errors.txt','a')
        f.write(f'{idx:06d}\t{i_path}\n')
        f.close()
        return None
    
    p_rotate = 0.85
    angles = list(np.arange(-15.0,15.0,0.1))
    
    # Rotation
    if random.random() < p_rotate:
        i_angle = random.sample(angles,1)[0]
        i_img = rotate(i_img,i_angle)
    
    i_transformed = transformer(image=i_img)['image']
    # return transformed
    
    # i_dst = cv2.resize(i_transformed,dsize=(dst_size,dst_size),interpolation=cv2.INTER_AREA)

    # i_dst = Image.fromarray(i_dst)
    # i_dst.save(i_path)
    i_dst = cv2.cvtColor(i_transformed,cv2.COLOR_RGB2BGR)
    cv2.imwrite(i_dstfpath,i_dst)
    return None
    
    

def generate_ds_number(src_path,dst_path,number_per_class,ballanced=True):
    
    #--- call transformer
    transformer = get_transformer()
    
    #--- get raw image pathinfo
    n_max = 0
    labels = []
    # dict_label_Ipaths = {}
    
    folders = os.listdir(src_path)
    folders.sort()
    
    for i_folder in folders:
        if i_folder[0] == '0':
            labels.append(i_folder)
            i_folder_path = os.path.abspath(os.path.join(src_path,i_folder))
            # dict_label_Ipaths[i_folder] = \
            i_src_paths = \
                [ os.path.join(i_folder_path,img_path) for img_path in os.listdir(i_folder_path)]
            
            print( i_folder, len(i_src_paths) )
            
            
            #--- generate ds | generate dataset with same number
            i_label = i_folder
            if i_label in ['02_AG1', '05_IWR']:
                continue
            print( src_path )
            print('Generate | ',i_label)
            n_target = number_per_class
            
            i_dst_path = os.path.join(dst_path,i_label)
            os.makedirs(i_dst_path,exist_ok=True)
            
            pathsidx = []
            for idx in range(n_target):
                pathsidx.append( (random.sample(i_src_paths,1)[0], idx) )

            func = partial(transform_img,transformer,i_dst_path)

            pool = Pool()
            with tqdm(total=n_target) as pbar:
                for _ in tqdm(pool.imap(func, pathsidx)):
                    pbar.update()
            pool.close()
            pool.join()
    
    return None
    


def generate_ds_factor(src_path,dst_path,factor,dst_size=256,ballanced=True):
    
    #--- call transformer
    transformer = get_transformer()
    
    #--- get raw image pathinfo
    n_max = 0
    labels = []
    dict_label_Ipaths = {}
    folders = os.listdir(src_path)
    folders.sort()
    for i_folder in folders:
        if i_folder[0] == '0':
            labels.append(i_folder)
            i_folder_path = os.path.abspath(os.path.join(src_path,i_folder))
            dict_label_Ipaths[i_folder] = \
                [ os.path.join(i_folder_path,img_path) for img_path in os.listdir(i_folder_path)]
            if n_max < len(dict_label_Ipaths[i_folder]):
                n_max = len(dict_label_Ipaths[i_folder])
            print( i_folder, len(dict_label_Ipaths[i_folder]) )
    labels.sort()
    
    #--- generate ds
    n_target = int(n_max*factor)
    
    for i_label in labels:
        print('Generate | ',i_label)
        
        i_src_paths = dict_label_Ipaths[i_label]
        i_dst_path = os.path.join(dst_path,i_label)
        os.makedirs(i_dst_path,exist_ok=True)
        
        func = partial(transform_img,transformer,i_src_paths,i_dst_path,dst_size)
        
        if ballanced:
            #--- generate ds | generate dataset with same number
            i_n_target = n_target
        else:
            #--- generate dataset with same number
            i_n_target = int(i_src_paths * factor)
        pool = Pool(90)
        with tqdm(total=i_n_target) as pbar:
            for _ in tqdm(pool.imap(func, range(n_target))):
                pbar.update()
        pool.close()
        pool.join()
    return None