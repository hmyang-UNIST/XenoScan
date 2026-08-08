import subprocess as subp
import random
import time
import glob
import sys
import os

def main(gpu_check=False):

    num_class = 3
    
    gpustat = os.popen('gpustat')
    outputs = gpustat.read()
    dict_gpus = {}
    for i in range(0,7):
        dict_gpus[i] = int( outputs.split('\n')[i+1].split('|')[2].split('/')[0] )
    
    gpu_ids = list( dict_gpus.keys() )

    command = {}
    idx = 0
    for network in ["resnet"]:
        for model_id in [1,2,3]:
            for timeblock in [1,2,3,4]:
                # model_tag = f'{date_of_work}_TB-{timeblock:02d}_832x512_20K_{network}_NCLS-{num_class:d}'
                model_tag = f'832x512_20K_{network}_NCLS-{num_class:03d}_TB-{timeblock:02d}_ID-{model_id:03d}'
                model_fname = f'M_Xenopus_{model_tag}.h5'
                M_CODE = f'M_Xenopus_{model_tag}'

                chpt_fpath = os.path.join(os.getcwd(),f'xx_chpt/{M_CODE:s}.hdf5')
                if os.path.isfile(chpt_fpath):
                    continue

                idx += 1
                cnt_try = 0
                while True:
                    cnt_try += 1
                    igpu = random.sample(gpu_ids,1)[0]
                    igpusize = 1024*10
                    if (dict_gpus[igpu] + igpusize) < (1024*22):
                        dict_gpus[igpu] += igpusize
                        command[idx] = f'python ./_train.py {igpu} {timeblock} {network} {num_class} {model_id}'
                        print(idx, command[idx], igpusize)
                        break
                    if cnt_try > 10:
                        break

    print(dict_gpus)
    if gpu_check:
        return None
    
    #--- run
    for idx in command.keys():
        subp.Popen(command[idx],shell=True)
        # time.sleep(1)
    return None

if __name__ == "__main__":
    main(gpu_check=False)