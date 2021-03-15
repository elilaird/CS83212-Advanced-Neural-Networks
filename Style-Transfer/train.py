import tensorflow as tf
from tensorflow import keras

print(f'Tensorflow version: {tf.__version__}')
print(f'Keras version: {keras.__version__}')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from io import BytesIO
import requests
#from tqdm import 
import copy

from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D, Conv2DTranspose

# Load VGG
pre_trained_model = tf.keras.applications.VGG19(include_top=False,
                                                      weights='imagenet')

def vgg_layers(inputs, target_layer):
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    if target_layer == 1:
        return x
    # Strides instead of maxpooling 
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', strides=2)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    if target_layer == 2:
        return x
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', strides=2)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    if target_layer == 3:
        return x
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4', strides=2)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    if target_layer == 4:
        return x
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4', strides=2)(x)
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    return x

def load_weights(trained_model, model):
    layer_names = [layer.name for layer in trained_model.layers]

    for layer in model.layers:
        b_name = layer.name.encode()
        if b_name in layer_names:
            layer.set_weights(trained_model.get_layer(b_name).get_weights())
            layer.trainable = False

def VGG19(trained_model, input_tensor=None, input_shape=None, target_layer=1):
    """
    VGG19, up to the target layer (1 for relu1_1, 2 for relu2_1, etc.)
    """
    if input_tensor is None:
        inputs = Input(shape=input_shape)
    else:
        inputs = Input(tensor=input_tensor, shape=input_shape)
    model = Model(inputs, vgg_layers(inputs, target_layer), name='vgg19')
    load_weights(trained_model, model)
    return model

target_layer = 3
vgg_model = VGG19(pre_trained_model, input_shape=(256, 256, 3), target_layer=target_layer)
vgg_model.summary()

def decoder_layers(inputs, layer):
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block5_conv1')(inputs)
    if layer == 5:
        return x

    
    #x = UpSampling2D((2, 2), name='decoder_block4_upsample')(x)
    x = Conv2DTranspose(1, kernel_size=(4,4), padding='same', strides=(2,2), name='decoder_block4_2DTrans')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block4_conv4')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='decoder_block4_conv1')(x)
    if layer == 4:
        return x

    #x = UpSampling2D((2, 2), name='decoder_block3_upsample')(x)
    x = Conv2DTranspose(1, kernel_size=(4,4), padding='same', strides=(2,2), name='decoder_block3_2DTrans')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv4')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='decoder_block3_conv1')(x)
    if layer == 3:
        return x

    #x = UpSampling2D((2, 2), name='decoder_block2_upsample')(x)
    x = Conv2DTranspose(1, kernel_size=(4,4), padding='same', strides=(2,2), name='decoder_block2_2DTrans')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='decoder_block2_conv2')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='decoder_block2_conv1')(x)
    if layer == 2:
        return x

    #x = UpSampling2D((2, 2), name='decoder_block1_upsample')(x)
    x = Conv2DTranspose(1, kernel_size=(4,4), padding='same', strides=(2,2), name='decoder_block1_2DTrans')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='decoder_block1_conv2')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='decoder_block1_conv1')(x)
    if layer == 1:
        return x

LAMBDA=1

def l2_loss(x):
    return K.sum(K.square(x)) / 2

class EncoderDecoder:
    def __init__(self, trained_model, input_shape=(256, 256, 3), target_layer=5,
                decoder_path=None):
        self.input_shape = input_shape
        print(self.input_shape)
        self.target_layer = target_layer
        self.trained_model = trained_model

        self.encoder = VGG19(self.trained_model, input_shape=self.input_shape, target_layer=target_layer)
    
        if decoder_path:
            self.decoder = load_model(decoder_path)
        else:
            self.decoder = self.create_decoder(target_layer)

        '''self.model = Sequential()
        self.model.add(self.encoder)
        self.model.add(self.decoder)'''
        self.model = Model(self.encoder.input, self.decoder(self.encoder.output))

        self.loss = self.create_loss_fn(self.encoder)

        #self.model.compile('adam', self.loss)

    def create_loss_fn(self, encoder):
        def get_encodings(inputs):
            encoder = VGG19(self.trained_model, inputs, self.input_shape, self.target_layer)
            return encoder.output

        def loss(img_in, img_out):
            encoding_in = get_encodings(img_in)
            encoding_out = get_encodings(img_out)
            return l2_loss(img_out - img_in) + \
                  LAMBDA*l2_loss(encoding_out - encoding_in)
        return loss

    def summary(self):
        self.model.summary()

    def create_decoder(self, target_layer):
        inputs = Input(shape=self.encoder.output_shape[1:])
        layers = decoder_layers(inputs, target_layer)
        output = Conv2D(3, (3, 3), activation='relu', padding='same',
                        name='decoder_out')(layers)
        return Model(inputs, output, name='decoder_%s' % target_layer)

    def export_decoder(self):
        self.decoder.save('decoder_%s.h5' % self.target_layer)

encoder_decoder = EncoderDecoder(pre_trained_model, target_layer=target_layer)
encoder_decoder.summary()

