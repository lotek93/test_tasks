import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import random
from tqdm import tqdm


base_dir = './'
train_clean_dir = base_dir + 'train/clean/'
train_noisy_dir = base_dir + 'train/noisy/'
val_clean_dir = base_dir + 'val/clean/'
val_noisy_dir = base_dir + 'val/noisy/'
size = 80


def load_files_list(directory):
    """ load files list from directory tree """
    files_list = []
    speakers = os.listdir(directory)
    for speaker in speakers:
        if os.path.isdir(directory + speaker):
            files = os.listdir(directory + speaker)
            for f in files:
                if os.path.isfile(directory + speaker + '/' + f):
                    files_list.append(directory + speaker + '/' + f)
    return files_list


def load_file(fname):
    """ load npy file with spectrogram, cut it on sizeXsize batch and return with label"""
    data = np.load(fname).T.astype(float)
    X, y = None, None
    if data.shape[1] >= size:
        X = data[0:size,0:size].reshape((1,size,size,1))
        for i in range(1, data.shape[1] // size):
            cut = data[0:size,i*size:(i+1)*size].reshape((1,size,size,1))
            X = np.concatenate((X, cut), axis=0)
        if 'clean' in fname:
            y = np.array([0 for i in range(X.shape[0])])
        elif 'noisy' in fname:
            y = np.array([1 for i in range(X.shape[0])])
        else:
            print('******** ERROR UNKNOWN LABEL *********')
    return X, y


def load_set(files):
    """ load dataset """
    split = 600
    big_loop = len(files) // split
    set_X, set_y = load_file(files[-1])
    for i in tqdm(range(big_loop)):
        tmp_X, tmp_y = load_file(files[i*split])
        for j in range(1, split):
            X, y = load_file(files[i*split+j])
            if X is not None:
                tmp_X = np.concatenate((tmp_X, X), axis=0)
                tmp_y = np.concatenate((tmp_y, y))
        set_X = np.concatenate((set_X, tmp_X), axis=0)
        set_y = np.concatenate((set_y, tmp_y))

    return set_X, set_y


def make_nn():
    model = tf.keras.Sequential([
        Conv2D(8, (3,3), input_shape=(size,size,1), activation='relu', padding='same'),
        MaxPool2D((2,2)),
        Conv2D(16, (3,3), activation='relu', padding='same'),
        MaxPool2D((2,2)),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPool2D((2,2)),
        Flatten(),
        Dropout(0.4),
        Dense(16, activation='relu'),
        Dropout(0.4),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


train_clean_files = load_files_list(train_clean_dir)
train_noisy_files = load_files_list(train_noisy_dir)
val_clean_files = load_files_list(val_clean_dir)
val_noisy_files = load_files_list(val_noisy_dir)
print(f'\nfound {len(train_clean_files)} clean train files, {len(train_noisy_files)} noisy train files,')
print(f'{len(val_clean_files)} clean val files, {len(val_noisy_files)} noisy val files')

# make shuffled train dataset
train_files = train_clean_files + train_noisy_files
random.shuffle(train_files)
print('\nloading train dataset')
train_X, train_y = load_set(train_files)
print(f'train set shape: {train_X.shape}')

# make shuffled val dataset
val_files = val_clean_files + val_noisy_files
random.shuffle(val_files)
print('\nloading val dataset')
val_X, val_y = load_set(val_files)
print(f'val set shape: {val_X.shape}')

model = make_nn()
model.summary()

early_stopping = EarlyStopping(patience=4, verbose=1)
reduce_lr = ReduceLROnPlateau(factor=0.2, patience=3, verbose=1)
h = model.fit(train_X, train_y, batch_size=64, epochs=30, validation_split=0.1, callbacks=[reduce_lr, early_stopping])

val_loss, val_accuracy = model.evaluate(val_X, val_y)
print(f'\nval loss = {val_loss}, val accuracy = {val_accuracy}')

print('\nsaving model into "classify.h5"')
model.save('classify.h5')
