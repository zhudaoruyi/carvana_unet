import cv2
import numpy as np
import pandas as pd
from generator import *
from sklearn.model_selection import train_test_split
from model.u_net import get_unet_128, get_unet_256, get_unet_512, get_unet_1024
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

df_train = pd.read_csv('input/train_masks.csv')
ids_train = df_train['img'].map(lambda s: s.split('.')[0])
ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))

#input_size = 256
#epochs     = 100
#batch_size = 16


def train_model(input_size, epochs, batch_size):
    if input_size == 128:
        model = get_unet_128()
    elif input_size == 256:
        model = get_unet_256()
    elif input_size == 512:
        model = get_unet_512()
    elif input_size == 1024:
        model = get_unet_1024()

    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=8,
                               verbose=1,
                               min_delta=1e-4),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=4,
                                   verbose=1,
                                   epsilon=1e-4),
                 ModelCheckpoint(monitor='val_loss',
                                 filepath='best_weights_256.h5py',
                                 save_best_only=True,
                                 save_weights_only=True),
                 TensorBoard(log_dir='logs')]

    model.fit_generator(generator=train_generator(ids_train_split, batch_size, input_size),
                        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
#                        steps_per_epoch=100,
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_generator(ids_valid_split, batch_size, input_size),
#                        validation_steps=50)
                        validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
    model.save('m25610016.h5')
train_model(256,100,16)
