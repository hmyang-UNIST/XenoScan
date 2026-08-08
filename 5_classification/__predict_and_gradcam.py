import os
import sys
import random

import cv2
import numpy as np
import pandas as pd

from _resnet import *
from tqdm import tqdm
import tensorflow as tf

'''
    # New version
    # Grad-CAM from global averaged pooling layer of ResNet18
'''
def get_gradCam_resnet(
    imgTensor,
    label_idx,
    model
    ):
    
    # imgTensor = img
    with tf.GradientTape() as tape:
        # inputs = image[np.newaxis, ...]
        inputs = imgTensor
        last_conv_layer_output = model.call_head(inputs)
        tape.watch(last_conv_layer_output)
        preds = model.call_tail(last_conv_layer_output)
        # top_pred_index = itf.argmax(preds[0])
        # top_class_channel = preds[:, top_pred_index]
        true_class_channel= preds[:, label_idx]

        grads = tape.gradient(true_class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    gradcam_resized = cv2.resize(heatmap, (512,832),cv2.INTER_CUBIC)
    # gradcams.append(gradcam)
    return gradcam_resized


def decode_predict( classes, prediction ):
    dict_out = {}
    for i in range( len(classes) ):
        i_prob = prediction[0][i]
        dict_out[classes[i]] = np.round(i_prob,4)
    return dict_out


def load_imgTensor( imgpath, apply_clahe=True ):
    bgr       = cv2.imread(imgpath)
    rgb       = cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
    
    if apply_clahe:
        CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        for ic in range(3):
            rgb[:,:,ic] = CLAHE.apply(rgb[:,:,ic])
    img       = np.round(rgb/255,5).astype(np.float32)
    imgTensor = img[np.newaxis,...]
    return imgTensor


def collect_gradCAM_resnet(rpath, classes, isid, iframe, imgpath, model, apply_clahe=True):
    
    imgTensor = load_imgTensor( imgpath, apply_clahe=apply_clahe )
    
    predict = model.predict(imgTensor,verbose=0)
    dict_predict = decode_predict( classes, predict )

    gradcam_paths = []
    for label_idx in range(len(classes)):
        gradcam_resized = get_gradCam_resnet(imgTensor,label_idx,model)
        
        # my_dpi=100
        # fig = plt.figure(figsize=(128*4/my_dpi, 208*4/my_dpi), dpi=my_dpi)

        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)

        # plt.imshow(imgTensor[0,:])
        # plt.imshow(gradcam_resized,alpha=0.45,cmap='jet')
        
        # gradcam_sum = np.uint8( gradcam_sum/n_layers*255 )
        # gradcam_sum_bgr = colorize(gradcam_sum)

        igradcam_fname = f'{isid:04d}_{iframe:04d}.png'
        irpath = os.path.join(rpath,f'__As_{classes[label_idx]}')
        os.makedirs(irpath,exist_ok=True)
        igradcam_fpath = os.path.join(irpath,igradcam_fname)
        
        gradcam_resized = (gradcam_resized*255).astype(np.uint8)
        gradcam_resized = cv2.cvtColor(gradcam_resized,cv2.COLOR_GRAY2BGR)
        # fig.savefig(igradcam_fpath, dpi=my_dpi)
        # plt.close()
        cv2.imwrite(igradcam_fpath,gradcam_resized)
        gradcam_paths.append(igradcam_fpath)
    return dict_predict, gradcam_paths


def main( argv ):
    gpuid = argv[1]
    import tensorflow as tf

    os.environ["CUDA_VISIBLE_DEVICES"]=f'{gpuid}'
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=1024*10)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
    
    IMG_HEIGHT = 208*4
    IMG_WIDTH  = 128*4
    network    = 'resnet'
    
    classes    = ['CONTROL', 'BIO', 'C59']
    num_class  = len(classes)
    
    # timeblocks = ['TB-01','TB-02','TB-03','TB-04']
    timeblocks = ['TB-01','TB-02','TB-03','TB-04']
    
    IDs = [0]
    
    # load models
    dict_models = {}
    for timeblock in timeblocks:
        dict_models[timeblock] = {}
        for ID in IDs:
            timeblock_code = int(timeblock.split('-')[-1])
            model_tag = \
                f'resnet18_TB-{timeblock_code:02d}'
            mpath = os.path.join('./models', f'{model_tag}.h5')
            print( os.path.isfile(mpath), mpath )
            # break

            model = ResNet18( num_class )
            model.build(input_shape = (None,IMG_HEIGHT,IMG_WIDTH,3))
            model.load_weights(mpath)
            dict_models[timeblock][ID] = model

            # mpath = glob.glob(f'./xx_chpt/{date_of_work}*{timeblock}*resnet_NCLS-003/*{timeblock}*-DENSE.h5')[0]
            # model = load_model(mpath)
            print( timeblock )
            print( mpath )

        # predict & get gradCAM

    dict_truelabelcode = {
        'CONTROL' : 0, 
        'BIO'     : 1, 
        'C59'     : 2,
    }
    dict_gradcam = {}
    dict_gradcam['CLAHE'] = []
    dict_gradcam['sid'] = []
    dict_gradcam['frame'] = []
    dict_gradcam['timeblock'] = []
    
    dict_gradcam['modelID'] = []
    
    dict_gradcam['truelabel'] = []
    dict_gradcam['labelcode'] = []
    dict_gradcam['prob_CONTROL'] = []
    dict_gradcam['prob_BIO'] = []
    dict_gradcam['prob_C59'] = []
    dict_gradcam['path_gCAM_as_CONTROL'] = []
    dict_gradcam['path_gCAM_as_BIO'] = []
    dict_gradcam['path_gCAM_as_C59'] = []
    dict_gradcam['srcpath'] = []

    # tf = pd.read_csv('./__testSet-input-gammaCorrection_v2023-04.csv')
    tf = pd.read_csv('./__testSet-input.csv')
    tf = tf[tf['timeblock'].isin(timeblocks)]
    
    N_imgs = len(tf)
    N_gpus = 7
    N_per_gpu = N_imgs // N_gpus
    
    if int(gpuid) == 6:
        itf = tf[N_per_gpu*int(gpuid):]
    else:
        itf = tf[N_per_gpu*int(gpuid):N_per_gpu*(1+int(gpuid))]
    
    for apply_clahe in [True,False]:
    
        for idx in tqdm(range( len(itf) )):
            
            isid = itf.iloc[idx]['sid']
            iframe = itf.iloc[idx]['frame']
            itimeblock = itf.iloc[idx]['timeblock']
            itruelabel = itf.iloc[idx]['label']
            itrue_code = dict_truelabelcode[itruelabel]

            imgpath = itf.iloc[idx]['imgpath']
            
            ifolder = imgpath.split('/')[-2]
            
            for ID in IDs:
                if apply_clahe:
                    rpath = os.path.join(f'./__testSet/__832x512/__GradCAM_{itimeblock}_ID-{ID:03d}_WITH-CLAHE',ifolder)
                else:
                    rpath = os.path.join(f'./__testSet/__832x512/__GradCAM_{itimeblock}_ID-{ID:03d}_NO-CLAHE',ifolder)

                imodel = dict_models[itimeblock][ID]
                dict_predict, gradcam_paths = \
                    collect_gradCAM_resnet(
                        rpath, classes, isid, iframe, imgpath, imodel, apply_clahe=apply_clahe)

                dict_gradcam['CLAHE'].append(apply_clahe)
                dict_gradcam['sid'].append(isid)
                dict_gradcam['frame'].append(iframe)
                dict_gradcam['timeblock'].append(itimeblock)
                
                dict_gradcam['modelID'].append(ID)
                
                dict_gradcam['truelabel'].append(itruelabel)
                dict_gradcam['labelcode'].append(itrue_code)
                dict_gradcam['prob_CONTROL'].append(dict_predict['CONTROL'])
                dict_gradcam['prob_BIO'].append(dict_predict['BIO'])
                dict_gradcam['prob_C59'].append(dict_predict['C59'])
                dict_gradcam['path_gCAM_as_CONTROL'].append(gradcam_paths[0])
                dict_gradcam['path_gCAM_as_BIO'].append(gradcam_paths[1])
                dict_gradcam['path_gCAM_as_C59'].append(gradcam_paths[2])
                dict_gradcam['srcpath'].append(imgpath)

    df = pd.DataFrame.from_dict(dict_gradcam)
    csvpath = f'./__testSet/__prediction-and-gradCAM-info_GPU-{gpuid}.csv'
    df.to_csv(csvpath,index=False)
    print(csvpath)
    return None

if __name__ == "__main__":
    main(sys.argv)