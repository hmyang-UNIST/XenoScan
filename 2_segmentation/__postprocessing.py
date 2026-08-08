## Imports
import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool


def msk_postprocessing( args ):
    
    impath_832 = args[0]
    impath_208 = args[1]
    isid       = args[2]
    iexp       = args[3]
    iplt       = args[4]
    iplt_id    = args[5]
    ilabelcode = args[6]
    ilabel     = args[7]
    iframe     = args[8]

    
    #--- merge mask information from both scale
    ipmap_832 = cv2.imread(impath_832,cv2.IMREAD_GRAYSCALE)
    ipmap_208 = cv2.imread(impath_208,cv2.IMREAD_GRAYSCALE)
    ipmap_208 = cv2.resize(ipmap_208,(512,832))
    
    for iblur in range(35):
        ipmap_832 = cv2.blur(ipmap_832,(7,7))
        ipmap_208 = cv2.blur(ipmap_208,(7,7))
    ipmap = np.where(ipmap_832 > ipmap_208, ipmap_832, ipmap_208)
        
    # if ipmap_832.max() > 0:
    #     ipmap_832 = (ipmap_832/ipmap_832.max()*255).astype(np.uint8)

    dstpath = './data/__rawdata/__v2023-04/__msks_blur_832'
    ifolder = f'{isid:04d}_{iexp}_{iplt}_{iplt_id}_{ilabelcode:02d}_{ilabel}'
    idstpath = os.path.join(dstpath,ifolder)
    os.makedirs(idstpath,exist_ok=True)
    ipath_msks_blur_832 = os.path.join(idstpath,f'{isid:04d}_{iframe:04}.png')
    # cv2.imwrite(ipath_msks_blur_832,ipmap_832)
    
    # if ipmap_208.max() > 0:
    #     ipmap_208 = (ipmap_208/ipmap_208.max()*255).astype(np.uint8)

    dstpath = './data/__rawdata/__v2023-04/__msks_blur_208'
    ifolder = f'{isid:04d}_{iexp}_{iplt}_{iplt_id}_{ilabelcode:02d}_{ilabel}'
    idstpath = os.path.join(dstpath,ifolder)
    os.makedirs(idstpath,exist_ok=True)
    ipath_msks_blur_208 = os.path.join(idstpath,f'{isid:04d}_{iframe:04}.png')
    # cv2.imwrite(ipath_msks_blur_208,ipmap_208)
    
    if ipmap.max() > 0:
        ipmap = (ipmap/ipmap.max()*255).astype(np.uint8)
    
    dstpath = './data/__rawdata/__v2023-04/__msks_blur_mgd'
    ifolder = f'{isid:04d}_{iexp}_{iplt}_{iplt_id}_{ilabelcode:02d}_{ilabel}'
    idstpath = os.path.join(dstpath,ifolder)
    os.makedirs(idstpath,exist_ok=True)
    ipath_msks_blur_mgd = os.path.join(idstpath,f'{isid:04d}_{iframe:04}.png')
    cv2.imwrite(ipath_msks_blur_mgd,ipmap)

    return ipath_msks_blur_832, ipath_msks_blur_208, ipath_msks_blur_mgd


def main():
    
    df = pd.read_csv('./__msk-prediction-info.csv')
    # df = df[:500]
    
    mpaths_832 = list(df['predpath_832x512'])
    mpaths_208 = list(df['predpath_208x128'])
    sids       = list(df['sid'])
    exps       = list(df['exp'])
    plts       = list(df['plt'])
    plt_ids    = list(df['plt_id'])
    labelcodes = list(df['labelcode'])
    labels     = list(df['label'])
    frames     = list(df['frame'])
    
    args = []
    for idx in range(len(mpaths_832)):
    # for idx in range(500):
        args.append(( mpaths_832[idx], mpaths_208[idx],
            sids[idx],exps[idx],plts[idx],plt_ids[idx],
            labelcodes[idx],labels[idx],frames[idx]))
    
    pool = Pool()
    mskpath_blur_832 = []
    mskpath_blur_208 = []
    mskpath_blur_mgd = []
    with tqdm(total=len(args)) as pbar:
        for imskpaths in tqdm(pool.imap(msk_postprocessing, args)):
            mskpath_blur_832.append(imskpaths[0])
            mskpath_blur_208.append(imskpaths[1])
            mskpath_blur_mgd.append(imskpaths[2])
            pbar.update()
    pool.close()
    pool.join()

    df.insert( 10,'mskpath_blur_832',mskpath_blur_832)
    df.insert( 11,'mskpath_blur_208',mskpath_blur_208)
    df.insert( 12,'mskpath_blur_mgd',mskpath_blur_mgd)
    # df.to_csv('./__msk-info.csv',index=False)

    return None


if __name__ == "__main__":
    main()