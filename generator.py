'''
    data generator
'''
import cv2
import numpy as np
from augmentation import *


#def train_generator(ids_train_split, batch_size, input_size):
#    while True:
#        x_batch = []
#        y_batch = []
#        for start in range(0, len(ids_train_split), batch_size):
#            end = min(start + batch_size, len(ids_train_split))
#            ids_train_batch = ids_train_split[start:end]
#            for id in ids_train_batch.values:
#                img = cv2.imread('/atlas/home/zwpeng/kaggle/train/{}.jpg'.format(id))
#                img = cv2.resize(img, (input_size, input_size))
#                mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
#                mask = cv2.resize(mask, (input_size, input_size))
#                img = randomHueSaturationValue(img,
#                                               hue_shift_limit=(-50, 50),
#                                               sat_shift_limit=(-5, 5),
#                                               val_shift_limit=(-15, 15))
#                img, mask = randomShiftScaleRotate(img, mask,
#                                                   shift_limit=(-0.0625, 0.0625),
#                                                   scale_limit=(-0.1, 0.1),
#                                                   rotate_limit=(-10, 10))
#                img, mask = randomHorizontalFlip(img, mask)
#                mask = np.expand_dims(mask, axis=2)
#                x_batch.append(img)
#                y_batch.append(mask)
#            x_batch = np.array(x_batch, np.float32) / 255.
#            y_batch = np.array(y_batch, np.float32) / 255.
#            yield x_batch, y_batch



def train_generator(ids_train_split, batch_size, input_size):
    while True:
        x_batch = []
        y_batch = []
        ids_train_batch = np.random.choice(ids_train_split.values, batch_size)
        for id in ids_train_batch:
            img = cv2.imread('/atlas/home/zwpeng/kaggle/train/{}.jpg'.format(id))
            img = cv2.resize(img, (input_size, input_size))
            mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (input_size, input_size))
            img = randomHueSaturationValue(img,
                                           hue_shift_limit=(-50, 50),
                                           sat_shift_limit=(-5, 5),
                                           val_shift_limit=(-15, 15))
            img, mask = randomShiftScaleRotate(img, mask,
                                               shift_limit=(-0.0625, 0.0625),
                                               scale_limit=(-0.1, 0.1),
                                               rotate_limit=(-10, 10))
            img, mask = randomHorizontalFlip(img, mask)
            mask = np.expand_dims(mask/255., axis=2)
            x_batch.append(img/255.)
            y_batch.append(mask)
        yield np.array(x_batch), np.array(y_batch)


#def valid_generator(ids_valid_split, batch_size, input_size):
#    while True:
#        for start in range(0, len(ids_valid_split), batch_size):
#            x_batch = []
#            y_batch = []
#            end = min(start + batch_size, len(ids_valid_split))
#            ids_valid_batch = ids_valid_split[start:end]
#            for id in ids_valid_batch.values:
#                img = cv2.imread('/atlas/home/zwpeng/kaggle/train/{}.jpg'.format(id))
#                img = cv2.resize(img, (input_size, input_size))
#                mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
#                mask = cv2.resize(mask, (input_size, input_size))
#                mask = np.expand_dims(mask, axis=2)
#                x_batch.append(img)
#                y_batch.append(mask)
#            x_batch = np.array(x_batch, np.float32) / 255.
#            y_batch = np.array(y_batch, np.float32) / 255.
#            yield x_batch, y_batch


def valid_generator(ids_valid_split, batch_size, input_size):
    while True:
        x_batch = []
        y_batch = []
        ids_valid_batch = np.random.choice(ids_valid_split.values, batch_size)
        for id in ids_valid_batch:
            img = cv2.imread('/atlas/home/zwpeng/kaggle/train/{}.jpg'.format(id))
            img = cv2.resize(img, (input_size, input_size))
            mask = cv2.imread('input/train_masks/{}_mask.png'.format(id), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (input_size, input_size))
            mask = np.expand_dims(mask/255., axis=2)
            x_batch.append(img/255.)
            y_batch.append(mask)
        yield np.array(x_batch), np.array(y_batch)


def plot_data(train=True):
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    df_train = pd.read_csv('input/train_masks.csv')
    ids_train = df_train['img'].map(lambda s: s.split('.')[0])
    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))

    if train:
        exam_X, exam_y = next(train_generator(ids_train_split, 16, 128))
    else:
        exam_X, exam_y = next(valid_generator(ids_valid_split, 16, 128))
    print(exam_X.shape, exam_y.shape)

    f,axes = plt.subplots(2,8,figsize=(20,5))
    ax = axes.flatten()
    for i,v in enumerate(exam_X):
        _ = ax[i].imshow(exam_X[i])

    fig,axes = plt.subplots(2,8,figsize=(20,5))
    ax = axes.flatten()
    for i,v in enumerate(exam_y):
        _ = ax[i].imshow(np.squeeze(exam_y[i]),cmap='gray')


