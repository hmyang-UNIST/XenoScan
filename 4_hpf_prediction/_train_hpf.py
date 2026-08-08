import os
import sys
import pandas as pd
import tensorflow as tf

# import tensorflow_addons as tfa  # unused; package is no longer maintained
from tensorflow.keras import optimizers

from _model import *
from _ds_loader import *

if __name__ == "__main__":
    '''
        input args
    '''
    gpuid = str( sys.argv[1] )
    
    '''
        setup box
    '''
    date_of_train = '221013'
    os.environ["CUDA_VISIBLE_DEVICES"]=gpuid
    
    BATCH_SIZE = 32
    steps_per_epoch_count = 100
    
    EPOCHS = 100000
    patience_earlyStop = 5*int(200000 / int(BATCH_SIZE*steps_per_epoch_count))
    
    model_info = f'__predict_HPF_MAE'
    mpath = './__models'
    os.makedirs(mpath,exist_ok=True)
    mname = f'{model_info}.h5'

    train_path = f'./__dataset/__trainSet_208x128/'
    valid_path = f'./__dataset/__validSet_208x128/'

    #--- load dataset
    IMG_HEIGHT = 208
    IMG_WIDTH  = 128

    train_ds = ds_generator(train_path,BATCH_SIZE)
    valid_ds = ds_generator(valid_path,BATCH_SIZE)

    #--- load lodel structure
    model = model_GAP_HPF(IMG_HEIGHT,IMG_WIDTH)
    
    #--- training initiate
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=patience_earlyStop,
            monitor='val_MAE',
            restore_best_weights=True
        ),
    ]

    model.compile(
        optimizers.Adam(learning_rate=1e-3),
        # loss=tf.keras.losses.MeanAbsolutePercentageError(),
        # loss=tf.keras.losses.MeanSquaredError(),
        loss = tf.keras.losses.MeanAbsoluteError(),
        metrics=["MAE"]
    )

    history = model.fit(train_ds,
                        steps_per_epoch  = steps_per_epoch_count,
                        validation_data  = valid_ds,
                        epochs=EPOCHS,
                        callbacks=callbacks )
    #--- save the model
    m_path = os.path.join(mpath,mname)
    model.save(m_path)
    print(m_path)

    #--- save training curve
    MAE        = list(history.history['MAE'])
    val_MAE    = list(history.history['val_MAE'])
    loss       = list(history.history['loss'])
    val_loss   = list(history.history['val_loss'])
    epochs_log = list(range(1, len(MAE) + 1))

    dict_train = {
        'steps'    : epochs_log,
        'MAE'      : MAE     ,
        'val_MAE ' : val_MAE ,
        'loss'     : loss    ,
        'val_loss' : val_loss
    }

    log_path = f'{mpath}/__training_log'
    os.makedirs(log_path,exist_ok=True)
    df_train = pd.DataFrame.from_dict(dict_train)
    df_fname = f'{model_info}_D-{date_of_train}.csv'
    df_spath = os.path.join(log_path,df_fname)

    df_train.to_csv(df_spath,index=False)
    print(df_spath)