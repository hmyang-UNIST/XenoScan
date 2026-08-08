import os
from _augmentation import *


'''
    # augmentation options
'''

if __name__ == "__main__":

    '''
        path info
    '''
    for i_tb in range(1,4+1):
        
        # train set
        src_path    = f'./DL_00_dataset/__SET-832x512/__centered_raw/01_TRAIN_RAW_832x512/TB-{i_tb:02d}'
        ds_dst_path = f'./DL_00_dataset/__SET-832x512/__train_832x512_20K/TB-{i_tb:02d}'
        os.makedirs(ds_dst_path,exist_ok=True)
        ballanced_set = True
        number_per_class = 200000
        generate_ds_number(src_path,ds_dst_path,number_per_class,ballanced=True)

        
        # test set
        src_path    = f'./DL_00_dataset/__SET-832x512/__centered_raw/02_BLIND_832x512/TB-{i_tb:02d}'
        ds_dst_path = f'./DL_00_dataset/__SET-832x512/__valid_832x512_20K/TB-{i_tb:02d}'
        os.makedirs(ds_dst_path,exist_ok=True)
        number_per_class = 25000
        ballanced_set = True
        generate_ds_number(src_path,ds_dst_path,number_per_class,ballanced=True)
