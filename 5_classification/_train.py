import os
import sys

from _resnet import *
from _model import *
from _evaluation import *
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import optimizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import tensorflow_addons as tfa
print(tf.__version__)

import matplotlib.pyplot as plt
import pandas as pd

'''
    SETUP BOX | START --------------
'''

def main(argv):

    gpu_id    = argv[1]      # gpu-id
    timeblock = int(argv[2]) # 1,2,3,4
    network   = argv[3]      # 'resnet' or 'dense'
    num_class = int(argv[4]) # 3 or 5
    model_id  = int(argv[5]) # 3 or 5
    
    '''
        training process initiate
    '''
    # gpu_size = dict_gpusize[(iroi,idpi)]
    gpu_size = 1024*10
    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        tf.config.experimental.set_virtual_device_configuration( gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=gpu_size)])
    except RuntimeError as e:
        print(e)
    # timeblock = 1
    
    if num_class == 3:
        target_classes = ['01_CONTROL', '03_BIO', '04_C59']
    if num_class == 5:
        target_classes = ['01_CONTROL', '02_AG1', '03_BIO', '04_C59', '05_IWR']
    num_class = len(target_classes)
    
    model_tag = f'832x512_20K_{network}_NCLS-{num_class:03d}_TB-{timeblock:02d}_ID-{model_id:03d}'
    model_fname = f'M_Xenopus_{model_tag}.h5'
    M_CODE = f'M_Xenopus_{model_tag}'
    
    '''
        PATH INFO
    '''
    train_path = f'./DL_00_dataset/__SET-832x512/__train_832x512_20K/TB-{timeblock:02d}'
    valid_path = f'./DL_00_dataset/__SET-832x512/__valid_832x512_20K/TB-{timeblock:02d}'
    
    model_path = f'./DL_01_Models/{model_tag}'
    os.makedirs(model_path,exist_ok=True)
    log_path   = f'./DL_02_Training_Log/{model_tag}'
    os.makedirs(log_path,exist_ok=True)
    fig_path   = f'./DL_03_Figures/{model_tag}'
    os.makedirs(fig_path,exist_ok=True)

    print(train_path)
    print(valid_path)
    
    #--- start
    BATCH_SIZE = 32
    IMG_HEIGHT = 832
    IMG_WIDTH  = 512
    
    # get model architecture
    if network == 'dense':
        # model = model_GAP( IMG_HEIGHT, IMG_WIDTH, num_class )
        model = model_DENSE( IMG_HEIGHT, IMG_WIDTH, num_class )
    elif network == 'resnet':
        # resnet architecture
        model = ResNet18( num_class )
        model.build(input_shape = (None,IMG_HEIGHT,IMG_WIDTH,3))

    steps_per_epoch = 250
    epochs = 5000
    patience_earlyStop = 375 # patience epochs
    
    #--- dataset
    #---generator object
    igen = ImageDataGenerator(
        rescale=1./255,
    )
    train_ds = igen.flow_from_directory(
        train_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        classes = target_classes)
    
    valid_ds = igen.flow_from_directory(
        valid_path,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        classes = target_classes)
    
    
    # config setups
    model.compile(
        loss='sparse_categorical_crossentropy',
        # optimizer=optimizers.RMSprop(lr=1e-4),
        # optimizer=optimizers.Adam(lr=1e-4),
        optimizer = "adam",
        # optimizer = optimizer,
        metrics=["accuracy"] )

    os.makedirs('xx_chpt',exist_ok=True)
    chpt_fpath = os.path.join(os.getcwd(),f'xx_chpt/{M_CODE:s}.hdf5')
    if os.path.isfile(chpt_fpath):
        model.load_weights(chpt_fpath)
        print('\n')
        print('model-training from last check point...')
        print('\n')
    
    model_checkpointer = keras.callbacks.ModelCheckpoint(
        filepath = chpt_fpath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose = 1
    )

    callbacks = [
        model_checkpointer,
        keras.callbacks.EarlyStopping(
            patience=patience_earlyStop,
            monitor='val_accuracy',
            restore_best_weights=True
        )
        # tfa.callbacks.TQDMProgressBar()
    ]

    # training the model
    history = model.fit(train_ds,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=valid_ds,
                        validation_steps=10,
                        callbacks=callbacks)

    #--- save model
    model_spath = os.path.join(model_path,model_fname)
    print(model_spath)

    model.save_weights(model_spath)

    #--- training curve
    acc        = list(history.history['accuracy'])
    val_acc    = list(history.history['val_accuracy'])
    loss       = list(history.history['loss'])
    val_loss   = list(history.history['val_loss'])
    epochs_log = list(range(1, len(acc) + 1))

    dict_train = {
        'acc' : acc     ,
        'val_acc ' : val_acc ,
        'loss' : loss    ,
        'val_loss' : val_loss,
        'steps' : epochs_log  
    }

    df_train = pd.DataFrame.from_dict(dict_train)
    df_fname = f'LOG_TrainCurve_{model_tag}.csv'
    df_spath = os.path.join(log_path,df_fname)
    
    df_train.to_csv(df_spath,index=False)
    print(df_spath)


if __name__ == "__main__":
    main(sys.argv)
