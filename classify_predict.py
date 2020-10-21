import tensorflow as tf
import numpy as np
import sys


def load_file(fname):
    size = 80
    data = np.load(fname).T.astype(float)
    X, y = None, None
    if data.shape[1] >= size:
        X = data[0:size,0:size].reshape((1,size,size,1))
        for i in range(1, data.shape[1] // size):
            cut = data[0:size,i*size:(i+1)*size].reshape((1,size,size,1))
            X = np.concatenate((X, cut), axis=0)
    return X


if len(sys.argv) == 2:
    print('\nloading model from "classify.h5"')
    model = tf.keras.models.load_model('classify.h5')

    batch = load_file(sys.argv[1])
    pred = model.predict(batch).mean(axis=0).argmax()

    if pred == 0:
        print(f'\nfile {sys.argv[1]} looks clean!')
    else:
        print(f'\nfile {sys.argv[1]} looks noisy!')
else:
    print('please specify .npy file name')
