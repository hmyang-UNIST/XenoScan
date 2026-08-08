import os
import pandas as pd
from skimage import io
from __quantification_sub import *

from multiprocessing import Pool
from tqdm import tqdm

def get_qinfo( imskpath ):

    imsk = io.imread(imskpath)    
    i_dict_qinfo = {}
    if imsk.max() == 0:
        i_dict_qinfo['state'] = 'empty'
        i_dict_qinfo['area'] = -1
        i_dict_qinfo['perimeter'] = -1
        i_dict_qinfo['length'] = -1
        i_dict_qinfo['circularity'] = -1
        return i_dict_qinfo
    else:
        try:
            iqinfo = RegOps(imsk).getRegGeomInfo()
            i_dict_qinfo['state'] = 'ok'
            i_dict_qinfo['area'] = iqinfo['area']
            i_dict_qinfo['perimeter'] = iqinfo['perim']
            i_dict_qinfo['length'] = iqinfo['length']
            i_dict_qinfo['circularity'] = iqinfo['circ']
        except:
            i_dict_qinfo['state'] = 'unknownError'
            i_dict_qinfo['area'] = -1
            i_dict_qinfo['perimeter'] = -1
            i_dict_qinfo['length'] = -1
            i_dict_qinfo['circularity'] = -1
        
    return i_dict_qinfo


def main():

    df = pd.read_csv('./__qinfo-input.csv')
    # df = df[:500]
    pool = Pool()
    
    dict_qinfo = {}
    dict_qinfo['exp'] = []
    dict_qinfo['label'] = []
    dict_qinfo['sid'] = []
    dict_qinfo['frame'] = []
    dict_qinfo['hpf'] = []
    dict_qinfo['state'] = []
    dict_qinfo['area'] = []
    dict_qinfo['perimeter'] = []
    dict_qinfo['length'] = []
    dict_qinfo['circularity'] = []
    dict_qinfo['roipath'] = []
    dict_qinfo['mskpath'] = []

    dict_qinfo['exp']     = list(df['exp'])
    dict_qinfo['label']   = list(df['label'])
    dict_qinfo['sid']     = list(df['sid'])
    dict_qinfo['frame']   = list(df['frame'])
    dict_qinfo['hpf']     = list( np.round(np.array(df['frame'])*25/60+2,2) )
    dict_qinfo['roipath'] = list(df['roipath'])
    dict_qinfo['mskpath'] = list(df['mskpath'])
    
    mskpaths = list(df['mskpath'])
    
    with tqdm(total=len(mskpaths)) as pbar:
        for i_dict_qinfo in tqdm(pool.imap(get_qinfo, mskpaths)):
            for ikey in i_dict_qinfo.keys():
                dict_qinfo[ikey].append( i_dict_qinfo[ikey])
            pbar.update()
    pool.close()
    pool.join()

    qdf = pd.DataFrame.from_dict( dict_qinfo )
    qdf.to_csv( './__qinfo-results.csv', index=False )

    return None

if __name__ == "__main__":
    main()