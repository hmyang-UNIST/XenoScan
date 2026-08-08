import os
import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

from skimage.io import imsave
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects
from skimage.registration import phase_cross_correlation
from skimage.measure import label, regionprops_table, regionprops


def fix_shift(img,shift):
    rows, cols = img.shape[:2]
    del_y, del_x = shift
    M = np.float32([[1,0,del_x],[0,1,del_y]])
    dst = cv2.warpAffine(img, M,(cols, rows))
    return dst


def get_mask_xywh(imgpath):
    img_bgr = cv2.imread(imgpath)
    # img_rgb = cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)
    try:
        img_bw = img_bgr[:,:,0] > threshold_otsu(img_bgr[:,:,0]) #blue
        img_bw = remove_small_objects(img_bw, 30000)
        img_bw = binary_fill_holes(img_bw)
        img_bw = img_bw.astype(np.uint8)
        cnt, _ = cv2.findContours(img_bw, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnt) == 1:
            x,y,w,h = cv2.boundingRect(cnt[0])
        else:
            # get x,y,w,h from all contours
            xmin = 500
            ymin = 1000
            xmax = 0
            ymax = 0
            for ic in cnt:
                ixmin = ic.min(axis=0)[0][0]
                iymin = ic.min(axis=0)[0][1]
                ixmax = ic.max(axis=0)[0][0]
                iymax = ic.max(axis=0)[0][1]
                if ixmin < xmin:
                    xmin = ixmin
                if iymin < ymin:
                    ymin = iymin
                if ixmax > xmax:
                    xmax = ixmax
                if iymax > ymax:
                    ymax = iymax
            x = xmin
            y = ymin
            w = xmax - xmin
            h = ymax - ymin
        
        # pad = 5
        # cropped_bgr = img_bgr[y-pad:y+h+pad,x-pad:x+w+pad,:]
        return x,y,w,h
    except:
        print(imgpath)
        return None


def process(isid):
    '''
        perform final step of preprocessing
        isid   : sample id
        return : None
        process results : micro-aligned & croped - padded sample images
    '''
    df = pd.read_csv('./__rawdata-info_v2023-04.csv')
    idf = df[df['sid'] == isid]
    iexp = idf.exp.unique()[0]
    iplt = idf.plt.unique()[0]
    iplt_id = idf['id-plt'].unique()[0]
    ilabel = idf.label.unique()[0]
    ilabelcode = idf.labelcode.unique()[0]

    #--- dst path setup
    dpath = './data/__rawdata/__v2023-04'
    
    #--- get reference image
    refpath = idf.iloc[0].rawpath
    ref = cv2.cvtColor(cv2.imread(refpath),cv2.COLOR_BGR2RGB)

    #--- get mask
    mask_rgb = np.zeros( ref.shape )
    pad = 5
    roi_xywh = get_mask_xywh(refpath)
    if roi_xywh == None:
        return -1
    else:
        x,y,w,h = roi_xywh
        mask_rgb[y-pad:y+h+pad,x-pad:x+w+pad,:]=1
        mask_gry = mask_rgb[:,:,0]
        # centering coordinates
        iy = y-pad
        ih = h+2*pad
        ix = x-pad
        iw = w+2*pad
        cy = iy + ih/2
        cx = ix + iw/2

    #--- do micro-alignment
    ref_roi = cv2.cvtColor(ref[:200,:,:],cv2.COLOR_RGB2GRAY)
    imgs_fixed = []
    for idx in tqdm(range( len(idf) )):
        iframe = idf.iloc[idx]['frame']
        if idx == 0:
            # reference image
            ipath = idf.iloc[idx]['rawpath']
            img = cv2.cvtColor(cv2.imread(ipath),cv2.COLOR_BGR2RGB)
            iimg_fixed = img
        else:
            ipath = idf.iloc[idx]['rawpath']
            img = cv2.cvtColor(cv2.imread(ipath),cv2.COLOR_BGR2RGB)
            img_roi = cv2.cvtColor(img[:200,:,:],cv2.COLOR_RGB2GRAY)
            img_updated = img_roi
            
            #--- micro-alignment
            del_y_total, del_x_total = 0,0
            while True:
                shift, error, eiffphase = phase_cross_correlation(ref_roi, img_updated, upsample_factor=10)
                del_y, del_x = shift
                del_y_total += del_y
                del_x_total += del_x
                if abs(del_y) < 1 and abs(del_x) < 1:
                    break
                img_updated = fix_shift(img_updated,shift)
            
            #--- collect shift information
            shift_total = (del_y_total,del_x_total)
            iimg_fixed = fix_shift(img,shift_total)
        
        # collect fixed images
        imgs_fixed.append( iimg_fixed )
        # get roi
        iroi_cropped = iimg_fixed[iy:iy+ih,ix:ix+iw,:]
        iih, iiw, _ = iroi_cropped.shape
        # centering cropped roi to 832x512 image
        img_new = np.zeros((832,512,3),dtype=np.uint8)
        iiy = int((832-iih)/2)
        iix = int((512-iiw)/2)
        img_new[iiy:iiy+iih,iix:iix+iiw,:] = iroi_cropped
        # save to idstpath
        idpath = os.path.join(dpath,f'{isid:04d}_{iexp}_{iplt}_{iplt_id}_{ilabelcode:02d}_{ilabel}')
        os.makedirs(idpath,exist_ok=True)
        idstpath = os.path.join(idpath,f'{isid:04d}_{iframe:04d}.jpg')
        cv2.imwrite(idstpath,cv2.cvtColor(img_new,cv2.COLOR_RGB2BGR))

    
    #--- visualize for check
    img_overlap = np.zeros( ref.shape, dtype=np.float32 )
    imgpaths = list(idf['rawpath'])
    for ipath in imgpaths:
        img = cv2.cvtColor(cv2.imread(ipath),cv2.COLOR_BGR2RGB)
        img_overlap += img/255
    img_overlap = img_overlap/len(imgpaths)
    
    img_overlap_after = np.zeros( ref.shape, dtype=np.float32 )
    for img in imgs_fixed:
        img_overlap_after += img/255
    img_overlap_after = img_overlap_after/len(imgpaths)
    

    #--- figpath
    figpath = './__process-check'
    os.makedirs(figpath,exist_ok=True)

    plt.figure(figsize=(12,8))
    plt.subplot(131)
    plt.imshow(ref)
    plt.title('Reference')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(img_overlap)
    plt.title('Overlap | before')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(img_overlap_after)
    plt.title('Overlap | after')
    plt.axis('off')
    ifigpath = os.path.join(figpath,f'{isid:04d}_{iexp}_{iplt}_{iplt_id}_{ilabelcode:02d}_{ilabel}.jpg')
    plt.savefig(ifigpath,dpi=150)
    plt.close()

    return None

from multiprocessing import Pool
from functools import partial
from tqdm import tqdm

def main():
    df = pd.read_csv('./__rawdata-info_v2023-04.csv')
    df = df[df['state'] == 'use-data']

    sids = list(df.sid.unique())
    for isid in tqdm(sids):
        process(isid)

    # pool = Pool(24)
    # with tqdm(total=len(sids)) as pbar:
    #     for _ in tqdm(pool.imap(process, sids)):
    #         pbar.update()

    # pool.close()
    # pool.join()
    return None

if __name__ == "__main__":
    main()