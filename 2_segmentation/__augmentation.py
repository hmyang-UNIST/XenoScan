import os
import glob
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
    dst = cv2.warpAffine(src, matrix, (width, height)).astype(np.uint8)
    return dst


# albumentation module
def get_transformer():
    
    transform = A.Compose([
        # A.Equalize         (mode='cv', by_channels=False, mask=None, mask_params=(), always_apply=False, p=0.15),
        A.ColorJitter      (brightness=0.2,contrast=0.2,saturation=0.02,hue=0.02,p=0.35),
        # A.RandomBrightness (limit=0.15, always_apply=False, p=0.35),
        # A.RandomContrast   (limit=0.15, always_apply=False, p=0.35),
        A.Flip             (p=0.75),
        A.CLAHE            (always_apply=False, clip_limit=(1, 10), tile_grid_size=(8, 8), p=1.0)
    ])
    
    return transform


def FLIP(img,msk):
    if random.random() < 0.5:
        img = cv2.flip(img,0)
        msk = cv2.flip(msk,0)
    else:
        img = cv2.flip(img,1)
        msk = cv2.flip(msk,1)
    return img,msk

#--- resize original dataset to 208x128 size
def resize_imwrite(dstpath,isrc):
    img = cv2.imread(isrc)
    img = cv2.resize(img,(128,208),cv2.INTER_AREA)
    ifname = isrc.split('/')[-1]
    idstpath = os.path.join(dstpath,ifname)
    cv2.imwrite(idstpath,img)
    return None


# v2023-04 - new sets | input = 832x512
def aug_single(roipaths,mskpaths,saveidx):

    # TRAIN SET
    roidst = './__dataset/__ds-trainValid-832x512/__roi/__jpgs'
    os.makedirs(roidst,exist_ok=True)
    mskdst = './__dataset/__ds-trainValid-832x512/__msk/__pngs'
    os.makedirs(mskdst,exist_ok=True)
    
    idx = random.sample( list( range(len(roipaths))) ,1)[0]

    ibgr = cv2.imread(roipaths[idx])
    irgb = cv2.cvtColor(ibgr,cv2.COLOR_BGR2RGB)
    
    #--- mask | filtering small dots around edges
    imsk = cv2.imread(mskpaths[idx])
    imsk = np.where( imsk > 128, 255, 0).astype(np.uint8)

    CLAHE_32 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(32,32))
    CLAHE_08 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    # for i in range(3):
    #     irgb[:,:,i] = CLAHE_32.apply(irgb[:,:,i])
    for i in range(3):
        irgb[:,:,i] = CLAHE_08.apply(irgb[:,:,i])
    irgb = cv2.bilateralFilter(irgb, 5, 75, 75)
    
    # Rotation
    p_rotate = 0.85
    angles = list(np.arange(-15.0,15.0,0.1))
    if random.random() < p_rotate:
        i_angle = random.sample(angles,1)[0]
        irgb = rotate(irgb,i_angle)
        imsk = rotate(imsk,i_angle)
    
    # irgb_transformed = transformer(image=irgb)['image']
    ibgr = cv2.cvtColor(irgb,cv2.COLOR_RGB2BGR)

    iroipath = os.path.join(roidst,f'{saveidx:06d}.jpg')
    cv2.imwrite(iroipath,ibgr)
    imskpath = os.path.join(mskdst,f'{saveidx:06d}.png')
    cv2.imwrite(imskpath,imsk)

    # resize and save
    roidst_small = './__dataset/__ds-trainValid-208x128/__roi/__jpgs'
    os.makedirs(roidst_small,exist_ok=True)
    mskdst_small = './__dataset/__ds-trainValid-208x128/__msk/__pngs'
    os.makedirs(mskdst_small,exist_ok=True)
    
    ibgr = cv2.resize(ibgr,(128,208),cv2.INTER_AREA)
    imsk = cv2.resize(imsk,(128,208),cv2.INTER_AREA)
    iroipath = os.path.join(roidst_small,f'{saveidx:06d}.jpg')
    cv2.imwrite(iroipath,ibgr)
    imskpath = os.path.join(mskdst_small,f'{saveidx:06d}.png')
    cv2.imwrite(imskpath,imsk)
    return None


def main():

    random.seed(1004)
    #--- train set
    roipaths = glob.glob('./__dataset/__trainValid-832x512-raw/__roi/*.jpg')
    roipaths.sort()
    mskpaths = glob.glob('./__dataset/__trainValid-832x512-raw/__msk/*.jpg')
    mskpaths.sort()
    len(roipaths), len(mskpaths)

    

    N_TRAIN = 100000

    func = partial(aug_single,roipaths,mskpaths)

    pool = Pool()
    with tqdm(total=N_TRAIN) as pbar:
        for _ in tqdm(pool.imap(func, range(N_TRAIN))):
            pbar.update()
    pool.close()
    pool.join()
    return None


if __name__ == "__main__":
    main()
    # main2()