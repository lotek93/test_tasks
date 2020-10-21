import tensorflow as tf
import numpy as np
import sys


def load_file(fname):
    data = np.load(fname).T.astype(float)
    length = data.shape[1]
    X = None
    if data.shape[1] >= size:
        X = data[0:size,0:size].reshape((1,size,size,1))
        for i in range(1, data.shape[1] // size):
            cut = data[0:size,i*size:(i+1)*size].reshape((1,size,size,1))
            X = np.concatenate((X, cut), axis=0)
        if data.shape[1] % size != 0:
            last_cut = data[0:size,data.shape[1]-size:].reshape((1,size,size,1))
            X = np.concatenate((X, last_cut), axis=0)
    return X, length


def unbatch(batch, length):
    spec = batch[0,:,:,0].reshape((size, size))
    for i in range(1,batch.shape[0]-1):
        cut = batch[i,:,:,0].reshape((size,size))
        spec = np.concatenate((spec, cut), axis=1)
    diff = length - spec.shape[1]
    last_cut = batch[-1,:,:,0].reshape((size,size))
    last_cut = last_cut[:,(size-diff):]
    spec = np.concatenate((spec, last_cut), axis=1)
    return spec


size = 80

print('spectrogram denoising')
if len(sys.argv) == 2:
    print('\nloading model from "denoising.h5"')
    model = tf.keras.models.load_model('denoising.h5')

    # fname = '/goznak/train/noisy/20/20_205_20-205-0004.npy'
    fname = sys.argv[1]
    data = np.load(fname).T.astype(float)
    batch, length = load_file(fname)
    pred = model.predict(batch)
    spec = unbatch(pred, length)

    shortname = fname.split('/')[-1]
    newname = shortname.split('.')
    newname.insert(1, '_clean.')
    newname = ''.join(newname)
    np.save(newname, spec.T.astype(np.float16))
    print(f'denoised file {newname} has been saved')

else:
    print('please specify .npy file name')
