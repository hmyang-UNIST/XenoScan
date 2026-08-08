import os
import sys
import time

import json
import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt
from multiprocessing import Pool

import cv2
import PIL.Image as Image
import glob

def fix_shift(img,shift):
    rows, cols = img.shape[:2]
    del_y, del_x = shift
    M = np.float32([[1,0,del_x],[0,1,del_y]])
    dst = cv2.warpAffine(img, M,(cols, rows))
    return dst


def rotate(src,angle):
    height, width, channel = src.shape
    matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
    dst = cv2.warpAffine(src, matrix, (width, height))
    return dst


def cut_plate_out(cut_arg):
    stime = time.time()
    crop_h, crop_w = 6500, 9000

    src_path  = cut_arg[0]
    dst_path  = cut_arg[1]
    shift     = cut_arg[2]
    crop_xya  = cut_arg[3]
    mask_path = cut_arg[4]
    
    # pass processed plate
    # if os.path.isfile(dst_path):
    #     return True
    i_x, i_y, i_a = crop_xya

    # read image
    # print(src_path,os.path.isfile(src_path))
    img = cv2.imread(src_path)

    # align image
    img = fix_shift(img,shift)

    # crop image
    img = img[i_y:i_y+crop_h,i_x:i_x+crop_w,:]

    # rotate image
    if i_a != 0:
        img = rotate(img,-i_a)

    # masking
    mask = cv2.imread(mask_path)
    # print(img.shape,mask.shape)
    
    masked = cv2.bitwise_and(img,mask)

    # save to destination
    cv2.imwrite(dst_path,masked)
    dtime = time.time() - stime
    # print(f'Time : {dtime:0.2f}s | saved : {dst_path}')
    return True


if __name__ == "__main__":
    
    '''
        # work-flow
            0. gathering dictionary info
            1. align (shift) the plates
            2. crop
            3. masking
            4. save
        # folder structure & naming format
            - C-xx_P-xx | folder
                - C-xx_P-xx_F-XXX.jpg | fname
    '''
    # source path
    spath = './data/01_rawdata/02_AsPlates/'
    
    # destination path
    dpath = './data/01_rawdata/03_crop_and_aligned'
    
    # mask path
    mpath = './mask_fit_9000x6500.png'
    
    # translational align value info
    with open('./plate_shift_info_v210807.json') as jf:
        dict_shift = json.load(jf)
    
    # crop information | after align
    
    
    cropsize = (6500,9000)

    cropinfo = open('./cropinfo_simple.txt','r')
    dict_crop_xya = {}
    for line in cropinfo:
        i_info = line.split(' ')
        i_code = i_info[0]
        i_x, i_y, i_angle = int(i_info[1]), int(i_info[2]), int(i_info[3])
        dict_crop_xya[i_code] = (i_x,i_y,i_angle)
    
    
    # generate argument for 'cut_plate_out'
    '''
        fix required
        C-05_P-01
        C-08_P-01
        C-08_P-02
        C-13_P-00
        C-13_P-02

    '''
    fix_target = ['C-08_P-01', 'C-08_P-02', 'C-13_P-02']
    cutover_args = []
    cnt = 0
    for i_key in dict_shift.keys():
        #--- cycle & plate info
        i_cyclePlate = i_key[:-3]
        if i_cyclePlate not in fix_target:
            continue
        
        i_cycle = i_cyclePlate.split('_')[0]  # C-XX
        i_plate = i_cyclePlate.split('_')[-1] # P-XX
        # if i_plate != 'P-02':
            # continue
        
        #--- create destination folder
        i_dst_folder = os.path.join(dpath,i_cyclePlate)
        os.makedirs(i_dst_folder,exist_ok=True)
        
        '''
            #--- calculate y_crop_adjust
                in 'old version' plate were generated dividing the whole scan into 3 part that equal sized
                in 'new version' plate were generated with 10% if margin in y-direction to avoid
        '''
        platepath_old = '/home/data/02_SSD4TB/hmyang/01_Xenopus/02_plates/01_raw_png'
        platepath_new = './data/01_rawdata/02_AsPlates'
        
        y_crop_adjust = 0
        if i_plate != 'P-00':
            frames_txt = list(dict_shift[i_key])
            frames = []
            for i_f_txt in frames_txt:
                frames.append(int(i_f_txt))
            frames.sort()
            FirstFrame = frames[0] # frame number as int
            
            firstplate_fname_old = f'{i_cycle}_{i_plate}_F-{FirstFrame:03d}.png'
            firstplate_fpath_old = os.path.join(platepath_old,firstplate_fname_old)
            # path check
            if os.path.isfile(firstplate_fpath_old):
                print('OLD |',firstplate_fpath_old,'o.k.')
            else:
                print('OLD |',firstplate_fpath_old,'check_path')
                sys.exit()
            firstplate_old = cv2.imread(firstplate_fpath_old)
            y_size_old = firstplate_old.shape[0]
            

            firstplate_fname_new = f'{i_cycle}_{i_plate}_F-{FirstFrame:03d}.jpg'
            firstplate_fpath_new = os.path.join(platepath_new,firstplate_fname_new)
            # path check
            if os.path.isfile(firstplate_fpath_new):
                print('NEW |',firstplate_fpath_new,'o.k.')
            else:
                print('NEW |',firstplate_fpath_new,'check_path')
                sys.exit()
            firstplate_new = cv2.imread(firstplate_fpath_new)
            y_size_new = firstplate_new.shape[0]

            y_size_diff = y_size_new - y_size_old
            if i_plate == 'P-01':
                y_crop_adjust = y_size_diff // 2
            else:
                y_crop_adjust = y_size_diff

        
        '''
        #--- generate argument for 'cut_plate_out'
            src_path  = cut_arg[0] # src image path
            dst_path  = cut_arg[1] # dst path
            shift     = cut_arg[2] # x,y, shift info | here we calculated shift info using small sized plate
                                     --> scale factor '10' should be multiplied
            crop_xya  = cut_arg[3] # crop point x,y, and angle a
            mask_path = cut_arg[4] # mask path
        '''
        for i_frame in dict_shift[i_key].keys():
            cnt += 1
            # path
            i_src_fname = f'{i_cyclePlate}_F-{int(i_frame):03d}.jpg'
            i_dst_fname = f'{i_cyclePlate}_F-{int(i_frame):03d}.jpg'
            
            i_src_fpath = os.path.join(spath,i_src_fname)
            i_dst_fpath = os.path.join(i_dst_folder,i_dst_fname)

            # shift info
            i_shift     = dict_shift[i_key][i_frame]
            i_del_y, i_del_x = i_shift
            i_del_y = i_del_y * 10 # scale factor
            i_del_x = i_del_x * 10 # scale factor
            i_shift = (i_del_y, i_del_x)
            
            # crop xya
            i_crop_xya  = dict_crop_xya[i_cyclePlate]
            i_x, i_y, i_a = i_crop_xya
            # i_y = i_y + y_crop_adjust
            i_crop_xya = (i_x, i_y, i_a)

            # mpath | mpath : './mask_fit_9000x6500.png'
            i_cutover_args = [i_src_fpath,i_dst_fpath,i_shift,i_crop_xya,mpath]
            cutover_args.append(i_cutover_args)
            
            # print(f'{cnt:04d}',i_key,i_cyclePlate,i_src_fpath,i_dst_fpath,i_shift,i_crop_xya)
    
    # run multiprocessing
    pool = Pool()
    with tqdm(total=len(cutover_args)) as pbar:
        for _ in tqdm(pool.imap_unordered(cut_plate_out, cutover_args)):
            pbar.update()
    
    pool.close()
    pool.join()
    
    