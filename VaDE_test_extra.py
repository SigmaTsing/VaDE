# -*- coding: utf-8 -*-

import argparse
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
import math
import numpy as np
import os
import scipy.io as scio
import theano
import theano.tensor as T
from utils import *
import warnings
warnings.filterwarnings("ignore")
             
def gmm_para_init(dataset: str):
    gmm_weights=scio.loadmat(os.path.join(
        'trained_model_weights', dataset + '_weights_gmm.mat'))
    u_init=gmm_weights['u']
    lambda_init=gmm_weights['lambda']
    theta_init=np.squeeze(gmm_weights['theta'])
    
    theta_p=theano.shared(
        np.asarray(theta_init,dtype=theano.config.floatX), 
        name="pi")
    u_p=theano.shared(
        np.asarray(u_init,dtype=theano.config.floatX),
        name="u")
    lambda_p=theano.shared(
        np.asarray(lambda_init,dtype=theano.config.floatX),
        name="lambda")
    return theta_p,u_p,lambda_p

def main(dataset: str):
    batch_size = 100
    latent_dim = 10
    intermediate_dim = [500,500,2000]
    theano.config.floatX='float32'
    X,Y = load_data(dataset)
    config = config_init(dataset)
    original_dim, _, n_centroid = config[:3]
    activation = config[-1]
    theta_p, u_p, lambda_p = gmm_para_init(dataset)

    x = Input(batch_shape=(batch_size, original_dim))
    h = Dense(intermediate_dim[0], activation='relu')(x)
    h = Dense(intermediate_dim[1], activation='relu')(h)
    h = Dense(intermediate_dim[2], activation='relu')(h)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)
    z = Lambda(Sampling(batch_size, latent_dim), 
               output_shape=(latent_dim,))([z_mean, z_log_var])
    h_decoded = Dense(intermediate_dim[-1], activation='relu')(z)
    h_decoded = Dense(intermediate_dim[-2], activation='relu')(h_decoded)
    h_decoded = Dense(intermediate_dim[-3], activation='relu')(h_decoded)
    x_decoded_mean = Dense(original_dim, activation=activation)(h_decoded)

    def get_gamma(tempz):
        temp_Z = T.transpose(K.repeat(tempz, n_centroid), [0,2,1])
        temp_u_tensor3 = T.repeat(u_p.dimshuffle('x', 0,1), batch_size, axis=0)
        temp_lambda_tensor3 = T.repeat(lambda_p.dimshuffle('x',0,1),
                                       batch_size,axis=0)
        temp_theta_tensor3 = theta_p.dimshuffle('x','x',0) \
            * T.ones((batch_size,latent_dim,n_centroid))
    
        temp_p_c_z = K.exp(K.sum(
            K.log(temp_theta_tensor3) \
            -0.5 * K.log(2 * math.pi * temp_lambda_tensor3) - \
            K.square(temp_Z - temp_u_tensor3) / (2 * temp_lambda_tensor3),
            axis=1))
        return temp_p_c_z/K.sum(temp_p_c_z,axis=-1,keepdims=True)

    p_c_z = Lambda(get_gamma, output_shape=(n_centroid,))(z_mean)
    #sample_output = Model(x, z_mean)
    p_c_z_output = Model(x, p_c_z)

    vade = Model(x, x_decoded_mean)
    vade.load_weights(os.path.join('trained_model_weights', dataset + '_nn.h5'))

    y_pred = np.argmax(p_c_z_output.predict(X, batch_size=batch_size), axis=1)
    print('Y_pred: ' + str(y_pred[:10]))
    print('Y: ' +  str(Y[:10]))
    accuracy, _ = cluster_acc(y_pred, Y)
    print('{} dataset VaDE - clustering accuracy: {:.2f}'
          .format(dataset, accuracy*100))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='test Fashion-MNIST or cifar-10-feature')
    parser.add_argument('dataset', 
                        choices=['fashion-mnist', 'cifar-10'], 
                        help='specify dataset')
    args = parser.parse_args()
    main(args.dataset)
