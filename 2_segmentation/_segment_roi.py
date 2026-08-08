## Imports
import os
import cv2
import numpy as np
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool


def get_segmented( args ):
    
    iroipath   = args[0]
    imskpath   = args[1]
    isid       = args[2]
    iexp       = args[3]
    iplt       = args[4]
    iplt_id    = args[5]
    ilabelcode = args[6]
    ilabel     = args[7]
    iframe     = args[8]

    
    #--- merge mask information from both scale
    imsk = cv2.imread(imskpath,cv2.IMREAD_GRAYSCALE)
    # if iframe < 68:
    #     th = 125
    # else:
    #     th = 50
    th = 75
    imsk = np.where( imsk > th, 1, 0).astype(np.uint8)

    cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(imsk)
    area = stats[1:,4]
    try:
        idx_max = np.argmax( area, axis=-1 ) + 1
        imsk = np.where(labels == idx_max, 255, 0)
    except:
        pass

    imsk = imsk.astype(np.uint8)
    
    dstpath = './data/__rawdata/__v2023-04/__msk_for_seg'
    ifolder = f'{isid:04d}_{iexp}_{iplt}_{iplt_id}_{ilabelcode:02d}_{ilabel}'
    idstpath = os.path.join(dstpath,ifolder)
    os.makedirs(idstpath,exist_ok=True)
    ipath_msk_for_seg = os.path.join(idstpath,f'{isid:04d}_{iframe:04}.png')
    cv2.imwrite(ipath_msk_for_seg,imsk)
    
    #--- roi segmentation
    ibgr = cv2.imread(iroipath)

    iseged = np.zeros_like(ibgr,dtype=np.uint8)
    for ic in range(3):
        iseged[:,:,ic] = np.where( imsk > 0, ibgr[:,:,ic], 0 )
    
    dstpath = './data/__rawdata/__v2023-04/__segmented_832x512'
    ifolder = f'{isid:04d}_{iexp}_{iplt}_{iplt_id}_{ilabelcode:02d}_{ilabel}'
    idstpath = os.path.join(dstpath,ifolder)
    os.makedirs(idstpath,exist_ok=True)
    ipath_segmented = os.path.join(idstpath,f'{isid:04d}_{iframe:04}.jpg')
    cv2.imwrite(ipath_segmented,iseged)


    # 208x128 size
    iseged_208 = cv2.resize(iseged,(128,208))
    dstpath = './data/__rawdata/__v2023-04/__segmented_208x128'
    ifolder = f'{isid:04d}_{iexp}_{iplt}_{iplt_id}_{ilabelcode:02d}_{ilabel}'
    idstpath = os.path.join(dstpath,ifolder)
    os.makedirs(idstpath,exist_ok=True)
    ipath_segmented_208 = os.path.join(idstpath,f'{isid:04d}_{iframe:04}.jpg')
    cv2.imwrite(ipath_segmented_208,iseged_208)


    return ipath_msk_for_seg, ipath_segmented, ipath_segmented_208


def main():
    
    df = pd.read_csv('./__msk-info.csv')
    # df = df[:500]
    
    roipath    = list(df['roipath'])
    mskpath    = list(df['mskpath_blur_mgd'])
    sids       = list(df['sid'])
    exps       = list(df['exp'])
    plts       = list(df['plt'])
    plt_ids    = list(df['plt_id'])
    labelcodes = list(df['labelcode'])
    labels     = list(df['label'])
    frames     = list(df['frame'])
    
    args = []
    for idx in range(len(roipath)):
    # for idx in range(500):
        args.append(( roipath[idx], mskpath[idx],
            sids[idx],exps[idx],plts[idx],plt_ids[idx],
            labelcodes[idx],labels[idx],frames[idx]))
    
    pool = Pool()
    path_msk_for_seg = []
    path_segmented = []
    path_segmented_208 = []
    with tqdm(total=len(args)) as pbar:
        for imskpaths in tqdm(pool.imap(get_segmented, args)):
            path_msk_for_seg.append(imskpaths[0])
            path_segmented.append(imskpaths[1])
            path_segmented_208.append(imskpaths[2])
            pbar.update()
    pool.close()
    pool.join()

    sf = pd.DataFrame()
    sf['sid'] = sids
    sf['exp'] = exps
    sf['plt'] = plts
    sf['plt_id'] = plt_ids
    sf['frame'] = frames
    sf['labelcode'] = labelcodes
    sf['label'] = labels
    sf['roipath'] = roipath
    sf['mskpath'] = path_msk_for_seg
    sf['segpath'] = path_segmented
    sf['segpath_208'] = path_segmented_208

    sf.to_csv('./__seged-info.csv',index=False)

    return None


if __name__ == "__main__":
    main()