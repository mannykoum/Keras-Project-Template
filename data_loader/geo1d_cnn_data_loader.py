from base.base_data_loader import BaseDataLoader
from data_loader.data_generator import DataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import transform, io
import numpy as np
import math # for floor

import os, re # for listdir, split

class GEO1dDataLoader(BaseDataLoader):
    def __init__(self, config):
        super(GEO1dDataLoader, self).__init__(config)

        'Create the training data generator'
        print('Creating the training data generator')
        tr_ids = os.listdir(self.config.data_loader.train_dir)
        tr_labels_str = list(map(self.label_image, tr_ids))
        tr_ids = list(map(self.png_to_id, tr_ids))
        tr_labels = dict(zip(tr_ids, tr_labels_str))
        self.train_generator = DataGenerator(tr_ids, tr_labels,
                                self.config.data_loader.train_dir,
                                batch_size=self.config.trainer.batch_size,
                                dim=[self.config.model.input_shape[0],self.config.model.input_shape[1]],
                                n_channels=1,
                                n_classes=self.config.model.output_shape,
                                shuffle=False)

        'Create the testing data generator'
        print('Creating the testing data generator')
        te_ids = os.listdir(self.config.data_loader.test_dir)
        te_labels_str = list(map(self.label_image, te_ids))
        te_ids = list(map(self.png_to_id, te_ids))
        te_labels = dict(zip(te_ids, te_labels_str))
        self.test_generator = DataGenerator(te_ids, te_labels,
                                self.config.data_loader.test_dir,
                                batch_size=self.config.trainer.batch_size,
                                dim=[self.config.model.input_shape[0],self.config.model.input_shape[1]],
                                n_channels=1,
                                n_classes=self.config.model.output_shape,
                                shuffle=False)

    def get_train_data(self):
        return self.train_generator

    def get_test_data(self):
        return self.test_generator

    def label_image(self, filename):
        fn_l = re.split('\W+',filename)
        angle = int(fn_l[len(fn_l)-2])
        angle = int(math.floor((angle%360)/10))
        return str(angle)

    def png_to_id(self, filename):
        return filename[0:len(filename)-4]

