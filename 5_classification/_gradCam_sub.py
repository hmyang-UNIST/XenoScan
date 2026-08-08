import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import cm

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import cv2

## use matplot jet for opencv
# source https://stackoverflow.com/questions/48003559/is-there-some-difference-between-colormap-of-opencv-and-matplotlib
def colorize(img):
    gray = None
    if img.ndim == 2:
        gray = img.copy()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canvas = np.uint8(cm.jet(gray)*255)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGBA2BGR)
    return canvas

'''
    # Old version
    # Grad-CAM from sum over each layer
'''
def get_gradcam_layer(
    imgTensor,
    label_idx,
    last_conv_layer_model,
    classifier_model ):
    
    with tf.GradientTape() as tape:
        # inputs = image[np.newaxis, ...]
        inputs = imgTensor
        last_conv_layer_output = last_conv_layer_model(inputs)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        # top_pred_index = tf.argmax(preds[0])
        # top_pred_index = tf.argmin(preds[0])
        # top_class_channel = preds[:, top_pred_index]
        true_class_channel= preds[:, label_idx]
        
        grads = tape.gradient(true_class_channel, last_conv_layer_output)
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # Average over all the filters to get a single 2D array
    gradcam = np.mean(last_conv_layer_output, axis=-1)
    # Clip the values (equivalent to applying ReLU)
    # and then normalise the values
    gradcam = np.clip(gradcam, 0, np.max(gradcam)) / np.max(gradcam)
    gradcam = cv2.resize(gradcam, (128,208))
    # gradcams.append(gradcam)
    
    return gradcam


def get_gradCAMs(label_idx,imgTensor,model):
    
    layer_names = [layer.name for layer in model.layers]
    layer_names = [layer_names[-3]]
    dict_layerCAM = {}
    
    for idx_layer in range( len(layer_names)):
        ilayer = layer_names[idx_layer]
        if (ilayer[:4] == 'conv') or (ilayer[:4] == 'acti'):
        # try:
            last_conv_layer = model.get_layer(ilayer)
            last_conv_layer_model = tf.keras.Model(model.inputs,last_conv_layer.output)

            classifier_input = tf.keras.Input( shape=last_conv_layer.output.shape[1:])
            classifier_layers = layer_names[idx_layer+1:]
            x = classifier_input
            for layer_name in classifier_layers:
                x = model.get_layer(layer_name)(x)
            classifier_model = tf.keras.Model(classifier_input, x)
            
            igradcam = get_gradcam_layer(
                imgTensor,
                label_idx,
                last_conv_layer_model,
                classifier_model )
            
            dict_layerCAM[ilayer] = igradcam
        # except:
        else:
            # print(idx_layer, layer_names[idx_layer])
            continue
    return dict_layerCAM


def collect_gradCAM(classes, imgpath, model):
    
    imgTensor, timeblock, truelabel, true_idx, ID, FRAME = load_imgTensor( imgpath )
    
    predict = model.predict(imgTensor)
    dict_predict = decode_predict( classes, predict )

    gradcam_paths = []
    for idx in range(3):
        dict_layerCam_true = get_gradCAMs(idx,imgTensor,model)
        
        gradcam_sum = np.zeros((208,128))
        n_layers = 0
        for ilayer in dict_layerCam_true.keys():
            if (ilayer[:4] == 'conv') or (ilayer[:4] == 'acti'):
                igradcam = dict_layerCam_true[ilayer]
                gradcam_sum += igradcam
                n_layers += 1
        
        my_dpi=100
        fig = plt.figure(figsize=(128/my_dpi, 208/my_dpi), dpi=my_dpi)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        plt.imshow(imgTensor[0,:])
        plt.imshow(gradcam_sum,alpha=0.75,cmap='jet')
        
        # gradcam_sum = np.uint8( gradcam_sum/n_layers*255 )
        # gradcam_sum_bgr = colorize(gradcam_sum)

        igradcam_fname = f'{ID:04d}_{FRAME:04d}.png'
        irpath = os.path.join(rpath,f'{truelabel}_{ID:03d}/__As_{classes[idx]}')
        os.makedirs(irpath,exist_ok=True)
        igradcam_fpath = os.path.join(irpath,igradcam_fname)

        fig.savefig(igradcam_fpath, dpi=my_dpi)
        plt.close()
        # cv2.imwrite(igradcam_fpath,gradcam_sum_bgr)
        gradcam_paths.append(igradcam_fpath)
    return dict_predict, gradcam_paths



