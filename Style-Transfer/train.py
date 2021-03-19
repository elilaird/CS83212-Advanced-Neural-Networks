import tensorflow as tf
from tensorflow import keras

print(f'Tensorflow version: {tf.__version__}')
print(f'Keras version: {keras.__version__}')

import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from io import BytesIO
import requests
import copy

import tensorflow_datasets as tfds
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess

from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, Model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D, Conv2DTranspose, Layer
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg_preprocess
from tensorflow.image import resize
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint



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

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    if target_layer == 2:
        return x
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', strides=2)(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    if target_layer == 3:
        return x
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4', strides=2)(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    if target_layer == 4:
        return x
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4', strides=2)(x)

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
    model = Model(inputs, vgg_layers(inputs, target_layer), name='vgg19', trainable=False)
    for layer in model.layers:
        layer.trainable = False
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

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam

LAMBDA=1

def l2_loss(x):
    return K.sum(K.square(x)) / 2

class EncoderDecoder:
    def __init__(self, trained_model, input_shape=(256, 256, 3), target_layer=5, decoder_path=None):
        self.input_shape = input_shape
        self.target_layer = target_layer
        self.trained_model = trained_model

        self.encoder = VGG19(self.trained_model, input_shape=self.input_shape, target_layer=target_layer)

        if decoder_path:
            self.decoder = load_model(decoder_path)
        else:
            self.decoder = self.create_decoder(target_layer)
   
        decoder_output = self.decoder(self.encoder.output)

        self.model = Model(self.encoder.input, decoder_output)

    def summary(self):
        self.model.summary()

    def create_decoder(self, target_layer):
        inputs = Input(shape=self.encoder.output.shape[1:])
        layers = decoder_layers(inputs, target_layer)
        output = Conv2D(3, (3, 3), activation='relu', padding='same',
                        name='decoder_out')(layers)
        decoder = Model(inputs, output, name='decoder_%s' % target_layer)
        return decoder

    def export_decoder(self):
        self.decoder.save('decoder_%s.h5' % self.target_layer)

encoder_decoder = EncoderDecoder(pre_trained_model, target_layer=target_layer)
encoder_decoder.summary()

# Create a dataset to train on (imagenette)
data_loader = tfds.load("imagenette", download=True)
train_ds, test_ds = data_loader['train'], data_loader['validation']
train_ds

from tensorflow.image import resize
train_ds, test_ds = data_loader['train'], data_loader['validation']

# Add batches and preprocessing 
BATCH_SIZE=16
n_train_observations = train_ds.cardinality().numpy()
n_test_observations = test_ds.cardinality().numpy()

steps_per_epoch = n_train_observations//BATCH_SIZE + n_train_observations%BATCH_SIZE
validation_steps = n_test_observations//BATCH_SIZE + n_test_observations%BATCH_SIZE

def preprocess(observation):
    img = observation['image']
    
    # Resize to target shape
    processed_img = resize(img, encoder_decoder.input_shape[:2])
    
    # vgg preprocess
    processed_img = vgg_preprocess(processed_img)
    
    # get the vgg encoding for the 'label'
    encoded_img = encoder_decoder.encoder(tf.expand_dims(processed_img, axis=0))
    encoded_img = tf.squeeze(encoded_img)

    return processed_img, processed_img #for encoder/decoder loss

train_ds = train_ds.map(
    lambda image: preprocess(image)).shuffle(1000).batch(BATCH_SIZE).repeat()
test_ds = test_ds.map(
    lambda image: preprocess(image)).shuffle(1000).batch(BATCH_SIZE).repeat()


print(f'Train observations: {n_train_observations}')
print(f'Test observations: {n_test_observations}')

from tensorflow.keras.losses import Loss

class CustLoss(Loss):
    def __init__(self, encoder, target_layer, LAMBDA=1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.target_layer = target_layer
        self.LAMBDA = LAMBDA
        
    def get_encodings(self, inputs):
        return self.encoder(inputs)
    
    def l2_loss(self, x):
        return K.sum(K.square(x)) / 2
        
    def call(self, img_in, img_out):
        encoding_in = self.get_encodings(img_in)
        encoding_out = self.get_encodings(img_out)
        return self.l2_loss(img_out - img_in) + self.LAMBDA*self.l2_loss(encoding_out-encoding_in)
    
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'encoder': self.encoder,
                'target_layer': self.target_layer, 'LAMBDA': self.LAMBDA}

adam = Adam(1e-4)
model = encoder_decoder.model
encoder = encoder_decoder.encoder
model.compile(loss=CustLoss(encoder, target_layer), optimizer=adam)

model.summary()

callbacks = [
    ModelCheckpoint(filepath='./encoder_decoder.h5'),
    ReduceLROnPlateau(monitor='val_loss', factor=.1, patience=5, min_lr=1e-6),
    EarlyStopping(patience=7)
]

history = model.fit(train_ds, validation_data=test_ds,
                    epochs=1000, callbacks=callbacks,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    verbose=2)