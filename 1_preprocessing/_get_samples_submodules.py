import os
import time
import json
import pandas as pd

import cv2
import numpy as np

import PIL.Image as Image
from tqdm import tqdm
from multiprocessing import Pool

def rotate90(src):
    img = Image.fromarray(src)
    img = img.rotate(90, expand=True)
    dst = np.asarray(img,dtype=np.uint8)
    # plt.imshow(im)
    return dst

def get_cropinfo(mpath):
    dict_sid_cropinfo = {}
    mask = cv2.imread(mpath)
    mask = mask.astype(dtype=np.uint8)
    contours, _ = cv2.findContours(mask[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    idx = 0
    box_w, box_h = 940,450

    for i in range(len(contours)-1,-1,-1):
        idx += 1
        i_contour = contours[i]
        x,y,w,h = cv2.boundingRect(i_contour)
        if w > h:
            w = box_w
            h = box_h
            i_rotate = True
        else:
            w = box_h
            h = box_w
            i_rotate = False
        dict_sid_cropinfo[idx] = (x,y,w,h,i_rotate)
    return dict_sid_cropinfo

def crop_samples(arg):
    stime = time.time()
    
    lpath = './02_samples/01_by_label/'
    ipath = './02_samples/02_by_id/'
    
    dict_path = {}
    dict_path['lpath'] = []
    dict_path['ipath'] = []
    
    df = arg[0]
    pcode = arg[1]
    ppath = arg[2]
    frame = arg[3]
    
    # read plate image
    img = cv2.imread(ppath)
    
    for i in range(len(df)):
        
        # crop each static position
        x,y = df.iloc[i]['cropx'],df.iloc[i]['cropy']
        w,h = df.iloc[i]['cropw'],df.iloc[i]['croph']
        
        i_img = img[y:y+h,x:x+w,:]
        
        if df.iloc[i]['cropr']:
            i_img = rotate90(i_img)
        
        # set fname | gf : global frame
        i_sid = df.iloc[i]['i_sid']
        i_gid = df.iloc[i]['i_gid']
        d_frm = df.iloc[i]['d_frm']
        i_label = df.iloc[i]['label']
        i_gf  = frame + d_frm
        
        i_fname = f'{i_gid:04d}_{i_gf:04d}.jpg'
        
        # save to 01_by_label
        i_lpath = os.path.join(lpath,i_label)
        i_fpath = os.path.join(i_lpath,i_fname)
        cv2.imwrite(i_fpath,i_img)
        
        # save to 02_by_id
        i_ifolder = f'{i_gid:04d}_{pcode}_sid-{i_sid:02d}_{i_label}'
        i_ipath = os.path.join(ipath,i_ifolder)
        os.makedirs(i_ipath,exist_ok=True)
        i_fpath = os.path.join(i_ipath,i_fname)
        cv2.imwrite(i_fpath,i_img)
    
    dtime = time.time() - stime
    # print(f'time consumed : {dtime:0.3f} sec')
    # return dict_path


def get_sample_info(arg):
    stime = time.time()
    
    lpath = './02_samples/01_by_label/'
    ipath = './02_samples/02_by_id/'
    
    dict_path = {}
    dict_path['lpath'] = []
    dict_path['ipath'] = []
    
    df = arg[0]
    pcode = arg[1]
    ppath = arg[2]
    frame = arg[3]
    
    for i in range(len(df)):
        
        # set fname | gf : global frame
        i_sid = df.iloc[i]['i_sid']
        i_gid = df.iloc[i]['i_gid']
        d_frm = df.iloc[i]['d_frm']
        i_label = df.iloc[i]['label']
        i_gf  = frame + d_frm
        
        i_fname = f'{i_gid:04d}_{i_gf:04d}.jpg'
        
        # save to 01_by_label
        i_lpath = os.path.join(lpath,i_label)
        i_fpath = os.path.join(i_lpath,i_fname)
        dict_path['lpath'].append(i_fpath)
        
        # save to 02_by_id
        i_ifolder = f'{i_gid:04d}_{pcode}_sid-{i_sid:02d}_{i_label}'
        i_ipath = os.path.join(ipath,i_ifolder)
        os.makedirs(i_ipath,exist_ok=True)

        i_fpath = os.path.join(i_ipath,i_fname)
        dict_path['ipath'].append(i_fpath)
    
    # dtime = time.time() - stime
    return dict_path
