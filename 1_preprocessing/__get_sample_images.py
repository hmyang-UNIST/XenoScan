import os
import sys
import time
import json
import pandas as pd

import cv2
import numpy as np

import PIL.Image as Image
from tqdm import tqdm
from multiprocessing import Pool

# from _get_samples_submodules import *

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
    box_w, box_h = 1024, 512

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
    
    # lpath = './02_samples/01_by_label/'
    # ipath = './02_samples/02_by_id/'
    
    dict_path = {}
    # dict_path['lpath'] = []
    dict_path['ipath'] = []
    
    df = arg[0]
    pcode = arg[1]
    ppath = arg[2]
    frame = arg[3]
    ipath = arg[4]
    
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
        # i_lpath = os.path.join(lpath,i_label)
        # i_fpath = os.path.join(i_lpath,i_fname)
        # cv2.imwrite(i_fpath,i_img)
        
        # save to 02_by_id
        i_ifolder = f'{i_gid:04d}_{pcode}_sid-{i_sid:02d}_{i_label}'
        i_ipath = os.path.join(ipath,i_ifolder)
        os.makedirs(i_ipath,exist_ok=True)
        i_fpath = os.path.join(i_ipath,i_fname)
        cv2.imwrite(i_fpath,i_img)
    
    # dtime = time.time() - stime
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



if __name__ == "__main__":

    basepath = './data/__rawdata'
    savepath = './data/__rawdata/__v2023-10'
    
    print('|--- Get samples start')
    
    mpath = './mask_fit_9000x6500-2023-10.png'
    dict_sid_cropinfo = get_cropinfo(mpath)
    
    # df_label = pd.read_csv('./00_label_info.csv')
    df_label = pd.read_csv('./00_label_info_v210806.csv')
    df_frame = pd.read_csv('./00_frame_info.csv')

    dict_frame = {}
    for i in range(len(df_frame)):
        i_cycle = df_frame.loc[i]['cycle']
        i_plate = df_frame.loc[i]['plate']
        i_del_frame = df_frame.loc[i]['del frame']

        i_key   = (i_cycle,i_plate)
        dict_frame[i_key] = i_del_frame

    dict_info = {}
    dict_label_to_id = {}
    dict_id_to_label = {}

    for i in range(len(df_label)):
        i_cycle = df_label.loc[i]['cycle']
        i_plate = df_label.loc[i]['plate']
        i_sidx  = df_label.loc[i]['static_index']
        i_gid   = df_label.loc[i]['global_index']
        i_label = df_label.loc[i]['label']
        i_ptype = df_label.loc[i]['phenotype']

        i_del_frame = dict_frame[(i_cycle,i_plate+1)]
        i_cropinfo  = dict_sid_cropinfo[i_sidx]

        dict_id_to_label[i_gid] = i_label

        if i_label not in dict_label_to_id.keys():
            dict_label_to_id[i_label] = []
            dict_label_to_id[i_label].append(i_gid)
        else:
            dict_label_to_id[i_label].append(i_gid)

        i_key = f'C-{i_cycle:02d}_P-{i_plate:02d}'
        if i_key not in dict_info.keys():
            dict_info[i_key] = {}
            dict_info[i_key]['i_sid'] = []
            dict_info[i_key]['i_gid'] = []
            dict_info[i_key]['label'] = []
            dict_info[i_key]['d_frm'] = []
            dict_info[i_key]['ptype'] = []
            dict_info[i_key]['cropx'] = []
            dict_info[i_key]['cropy'] = []
            dict_info[i_key]['cropw'] = []
            dict_info[i_key]['croph'] = []
            dict_info[i_key]['cropr'] = []
        else:

            dict_info[i_key]['i_sid'].append(i_sidx)
            dict_info[i_key]['i_gid'].append(i_gid)
            dict_info[i_key]['label'].append(i_label)
            dict_info[i_key]['d_frm'].append(i_del_frame)
            dict_info[i_key]['ptype'].append(i_ptype)
            dict_info[i_key]['cropx'].append(i_cropinfo[0])
            dict_info[i_key]['cropy'].append(i_cropinfo[1])
            dict_info[i_key]['cropw'].append(i_cropinfo[2])
            dict_info[i_key]['croph'].append(i_cropinfo[3])
            dict_info[i_key]['cropr'].append(i_cropinfo[4])    

    print('|--- collect info | done.')
    
    ppath = f'{basepath}/01_rawdata/03_crop_and_aligned/'
    plate_codes = os.listdir(ppath)
    plate_codes.sort()
    # print( plate_codes[:3] )

    # prepare folders
    # lpath = f'{basepath}/02_samples/01_by_label/'
    # for i_label in dict_label_to_id.keys():
    #     os.makedirs(os.path.join(lpath,i_label),exist_ok=True)
    ipath = f'{savepath}/_sampleImges_by_id/'
    
    # generate args for "crop_samples"
    args = []
    NA_pcodes = []
    # NA_pcodes = ['C-01_P-02', 'C-05_P-00'] # not a single sequience..
    plate_codes = ['C-09_P-02']

    dict_samplepath = {}
    for i_pcode in tqdm(plate_codes):
        if i_pcode in NA_pcodes:
            continue
        df = pd.DataFrame.from_dict(dict_info[i_pcode])
        p_fnames = os.listdir(os.path.join(ppath,i_pcode))
        p_fnames.sort()
        # print(p_fnames)

        dict_samplepath[i_pcode] = []

        for i_frame in range(len(p_fnames)):
            i_p_fname = p_fnames[i_frame]
            i_p_fpath = os.path.join(ppath,i_pcode,i_p_fname)
            i_arg = (df,i_pcode,i_p_fpath,i_frame,ipath)

            # i_dict_path = get_sample_info(i_arg)
            # dict_samplepath[i_pcode].append(i_dict_path)
            args.append(i_arg)
    
    
    all_paths = []
    for i_key in dict_samplepath.keys():
        i_dicts = dict_samplepath[i_key]
        for i_dict in i_dicts:
            all_paths.extend(i_dict['ipath'])

    all_paths.sort()

    print('|--- generate path info | done.')
    
    # '''
    #     #--- generate datainfo csv for all sample image files
    # '''
    # dict_datainfo = {}
    # dict_datainfo['id_global'] = []
    # dict_datainfo['frame'] = []
    # dict_datainfo['label'] = []
    # dict_datainfo['phenoStatus'] = []
    # dict_datainfo['cycle'] = []
    # dict_datainfo['plate'] = []
    # dict_datainfo['id_static'] = []
    # dict_datainfo['path_by_id'] = []
    # dict_datainfo['path_by_label'] = []

    # for i_path in tqdm(all_paths):
    #     i_info = i_path.split('/')[-2:]
    #     # i_info[0] : 0002_C-01_P-01_sid-07_01-CONTROL
    #     # i_info[1] : 0002_0019.jpg

    #     id_global = int(i_info[0][:4])
    #     frame     = int(i_info[1][5:5+4])
    #     label     = i_info[0].split('_')[-1]
    #     cycle     = i_info[0].split('_')[1][-2:]
    #     plate     = i_info[0].split('_')[2][-2:]
    #     id_static = i_info[0].split('_')[3][-2:]
    #     path_by_id= i_path
    #     path_by_label = os.path.join(lpath,label,i_info[1])

    #     phynoStatus = df_label[df_label['global_index'] == id_global].phenotype_v2.iloc[0]

    #     dict_datainfo['id_global'].append(id_global)
    #     dict_datainfo['frame'].append(frame)
    #     dict_datainfo['label'].append(label)
    #     dict_datainfo['phenoStatus'].append(phynoStatus)
    #     dict_datainfo['cycle'].append(cycle)
    #     dict_datainfo['plate'].append(plate)
    #     dict_datainfo['id_static'].append(id_static)
    #     dict_datainfo['path_by_id'].append(path_by_id)
    #     dict_datainfo['path_by_label'].append(path_by_label)

    # df_datainfo = pd.DataFrame.from_dict(dict_datainfo)
    # df_datainfo[:3]

    # df_datainfo.to_csv('./00_datainfo_v210807.csv',index=False)
    
    # sys.exit()
    
    pool = Pool()
    with tqdm(total=len(args)) as pbar:
        for _ in tqdm(pool.imap_unordered(crop_samples, args)):
            pbar.update()

    pool.close()
    pool.join()


