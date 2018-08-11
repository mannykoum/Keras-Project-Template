from base.base_data_loader import BaseDataLoader
from data_loader.data_generator import DataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, \
                                      img_to_array, load_img
from skimage import transform, io
import numpy as np
import math # for floor

import os, re # for listdir, split

class GEO1dDataLoader2(BaseDataLoader):
    def __init__(self, config):
        super(GEO1dDataLoader2, self).__init__(config)
        datagen = ImageDataGenerator(
            featurewise_center=False, 
            samplewise_center=False, 
            featurewise_std_normalization=False, 
            samplewise_std_normalization=False, 
            zca_whitening=False, 
            zca_epsilon=1e-06, 
            rotation_range=0.0, 
            width_shift_range=0.0, 
            height_shift_range=0.0, 
            brightness_range=None, 
            shear_range=0.0, 
            zoom_range=0.0, 
            channel_shift_range=0.0, 
            fill_mode='nearest', 
            cval=0.0, 
            horizontal_flip=False, 
            vertical_flip=False, 
            rescale=1./255,
            preprocessing_function=None, 
            data_format=None, 
            validation_split=0.0)

        tgt_size = (self.config.data_loader.input_shape[1],
                        self.config.data_loader.input_shape[2])
        
        'Create the training data generator'
        print('Creating the training data generator')
        self.train_generator = datagen.flow_from_directory(
                                self.config.data_loader.train_dir,
                                target_size=tgt_size,
                                color_mode='grayscale',
                                batch_size=self.config.trainer.batch_size,
                                class_mode='sparse')

        'Create the testing data generator'
        print('Creating the testing data generator')
        self.test_generator = datagen.flow_from_directory(
                                self.config.data_loader.test_dir,
                                target_size=tgt_size,
                                color_mode='grayscale',
                                batch_size=self.config.trainer.batch_size,
                                class_mode='sparse')

    def get_train_data(self):
        return self.train_generator

    def get_test_data(self):
        return self.test_generator

