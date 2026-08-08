import os
import numpy as np
import cv2
import PIL.Image as Image
import matplotlib.pyplot as plt
import datetime

import pandas as pd
import glob
import shutil

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool

def get_plates(ipathinfo):
    
    pltpath = './01_rawdata/02_AsPlates'
    pltpath_small = './01_rawdata/02-1_AsPlates_small'
    
    # print(pathinfo)
    img_path, i_cycle, i_frame = ipathinfo
    i_img = cv2.imread(img_path)
    size_y,size_x,_ = i_img.shape
    plate_size_y    = int(size_y/3)
    plate_margin    = int(plate_size_y / 20 )

    plates = []
    plates.append( i_img[:plate_size_y+plate_margin,:,:] )
    plates.append( i_img[plate_size_y-plate_margin:plate_size_y*2+plate_margin,:,:] )
    plates.append( i_img[plate_size_y*2-plate_margin:,:,:] )
    
    for idx in range(3):
        i_plate = plates[idx]
        #--- save original resolution
        i_plt_fname = f'C-{i_cycle:02d}_P-{idx:02d}_F-{i_frame:03d}.jpg'
        i_plt_fpath = os.path.join(pltpath,i_plt_fname)
        cv2.imwrite(i_plt_fpath,i_plate)

        #--- save small size
        i_plate = cv2.resize(i_plate, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)
        i_plt_fpath = os.path.join(pltpath_small,i_plt_fname)
        cv2.imwrite(i_plt_fpath,i_plate)

        
if __name__ == "__main__":
    rawpath = './01_rawdata/01_AsScanned/'
    folders = os.listdir(rawpath)
    folders.sort()
    
    jpgpaths = glob.glob(rawpath+'*/*.jpg')
    jpgpaths.sort()

    dict_cycle_imgpath = {}
    cnt_img = {}
    for i_path in jpgpaths:
        i_folder = i_path.split('/')[-2]
        i_fname  = i_path.split('/')[-1]
        i_info = i_folder.split('_')
        i_cycle = int(i_info[1][5:])
        # print(i_folder, i_cycle)
        # break
        try:
            dict_cycle_imgpath[i_cycle][0]
        except:
            cnt_img[i_cycle] = 0
            dict_cycle_imgpath[i_cycle] = []
        i_day   = int(i_info[-1][-1])

        #--- plate 03 | IWR
        if (i_cycle == 2) and i_day in [5,6]:
            i_cycle = 22
            try:
                dict_cycle_imgpath[i_cycle][0]
            except:
                cnt_img[i_cycle] = 0
                dict_cycle_imgpath[i_cycle] = []

        if i_fname.split('.')[-1] == 'jpg':
            cnt_img[i_cycle] += 1
            dict_cycle_imgpath[i_cycle].append(i_path)

    n_scan = 0
    for ic in dict_cycle_imgpath.keys():
        n_scan += len(dict_cycle_imgpath[i_cycle])
    n_scan

    pltpath = './01_rawdata/02_AsPlates'
    os.makedirs(pltpath,exist_ok=True)

    pltpath_small = './01_rawdata/02-1_AsPlates_small'
    os.makedirs(pltpath_small,exist_ok=True)


    dict_info = {}
    dict_info['cycle']    = []
    dict_info['frame']    = []
    dict_info['scantime'] = []
    dict_info['info']     = []
    dict_info['plate']    = []
    dict_info['fpath']    = []
    dict_info['rawpath']  = []

    pathinfo = []

    for i_cycle in dict_cycle_imgpath.keys():

        # if i_cycle == 22:
        #     continue

        i_fpaths = dict_cycle_imgpath[i_cycle]
        for i_frame in tqdm(range(len(i_fpaths))):
            i_fpath = i_fpaths[i_frame]
            i_info = i_fpath.split('/')[-2]

            i_mtime = os.path.getmtime(i_fpath)
            i_mtime = datetime.datetime.fromtimestamp(i_mtime)
            
            # i_fname_cycle = f'C-{i_cycle:02d}_'
            # i_fname_frame = f'F-{i_frame:03d}'

            if i_cycle == 22:

                idx = 2
                i_fname_header = f'C-{2:02d}_P-{idx:02d}_F-{i_frame+len(dict_cycle_imgpath[2]):03d}'
                i_plt_fname = f'{i_fname_header}.jpg'
                i_plt_fpath = os.path.join(pltpath,i_plt_fname)

                dict_info['cycle'].append(i_cycle)
                dict_info['frame'].append(i_frame+len(dict_cycle_imgpath[2]))
                dict_info['scantime'].append(i_mtime)
                dict_info['info'].append(i_info)
                dict_info['plate'].append(idx)
                dict_info['fpath'].append(i_plt_fpath)
                dict_info['rawpath'].append(i_fpath)
                
                # continue
                # save original size
                # shutil.copy(i_fpath,i_plt_fpath)

                # # save small size
                # i_img = cv2.imread(i_fpath)
                # i_img = cv2.resize(i_img, dsize=(0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_LINEAR)

                # # i_plt_fname = f'{i_fname_header}P-{idx:02d}.jpg'
                # i_plt_fpath = os.path.join(pltpath_small,i_plt_fname)

                # cv2.imwrite(i_plt_fpath,i_img)
            else:
                
                # continue
                pathinfo.append( (i_fpath,i_cycle,i_frame) )
                
                for idx in range(3):
                    i_fname_header = f'C-{i_cycle:02d}_P-{idx:02d}_F-{i_frame:03d}'
                    # pathinfo.append([i_fpath,i_fname_header])
                    i_plt_fname = f'{i_fname_header}.jpg'
                    i_plt_fpath = os.path.join(pltpath,i_plt_fname)

                    dict_info['cycle'].append(i_cycle)
                    dict_info['frame'].append(i_frame)
                    dict_info['scantime'].append(i_mtime)
                    dict_info['info'].append(i_info)
                    dict_info['plate'].append(idx)
                    dict_info['fpath'].append(i_plt_fpath)
                    dict_info['rawpath'].append(i_fpath)


    df_info = pd.DataFrame.from_dict(dict_info)
    df_fpath = 'xx_plate_info.csv'
    # df_fpath = os.path.join('./',df_fname)
    df_info.to_csv(df_fpath,index=False)

    pool = Pool()
    with tqdm(total=len(pathinfo)) as pbar:
        for _ in tqdm( pool.imap_unordered(get_plates,pathinfo) ):
            pbar.update()
    pool.close()
    pool.join()