#!/usr/bin/env python3

import os, psutil, math

import tensorflow as tf
from keras import backend as k

# For loading data
from keras.preprocessing.image import ImageDataGenerator, array_to_img, \
                                      img_to_array, load_img
from skimage import transform, io
import numpy as np

# For creating the model
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model

# For training and callbacks
from keras.callbacks import ModelCheckpoint, TensorBoard

# Debugging flag
debug = True
exp_name = 'GEO1d_cnn'
###################################
# TensorFlow wizardry
tf_config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
tf_config.gpu_options.allow_growth = True
 
# Only allow a fraction of the GPU memory to be allocated
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=tf_config))
run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

### Load data 
datagen = ImageDataGenerator(featurewise_center=False, 
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

tgt_size = (256,256)
train_dir = "/home/mannykoum/Manos_repos/tf_tutorial/GEO1d/data_labeled/train/"
test_dir = "/home/mannykoum/Manos_repos/tf_tutorial/GEO1d/data_labeled/valid/"
batch_size = 16 

print('Creating the training data generator')
train_generator = datagen.flow_from_directory(
                        directory=test_dir,
                        target_size=tgt_size,
                        color_mode='grayscale',
                        batch_size=batch_size,
                        class_mode='sparse')

print('Creating the testing data generator')
test_generator = datagen.flow_from_directory(
                        directory=test_dir,
                        target_size=tgt_size,
                        color_mode='grayscale',
                        batch_size=batch_size,
                        class_mode='sparse')

### Build the model architecture
input_shape = (256, 256, 1)
output_shape = 36

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', 
                 input_shape=input_shape))
#model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(output_shape, activation='sigmoid'))

# if the config has the debug flag on, turn on tfdbg (TODO: make it work)
print("Before multi:")
if (debug == True):
    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, 
            show_layer_names=True)

# TODO: check num_gpus
# On model saving
# To save the multi-gpu model, use .save(fname) or .save_weights(fname) 
# with the template model (the argument you passed to multi_gpu_model), 
# rather than the model returned by multi_gpu_model.
model = multi_gpu_model(model, gpus=2)

print("After multi:")
if (debug == True):
    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, 
            show_layer_names=True)

model.compile(
      optimizer='adadelta',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'], 
      loss_weights=None,
      sample_weight_mode=None,
      weighted_metrics=None,
      target_tensors=None)
      #options=run_opts)

# model_checkpoint_dir = ''
callbacks = []
# callbacks.append(
#             ModelCheckpoint(
#                 filepath=os.path.join(model_checkpoint_dir, 
#                     '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % exp_name),
#                 monitor='val_loss',
#                 mode=self.config.callbacks.checkpoint_mode,
#                 save_best_only=self.config.callbacks.checkpoint_save_best_only,
#                 save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
#                 verbose=self.config.callbacks.checkpoint_verbose,
#             )
#         )

# self.callbacks.append(
#             TensorBoard(
#                 log_dir=self.config.callbacks.tensorboard_log_dir,
#                 write_graph=self.config.callbacks.tensorboard_write_graph,
#             )
#         )


num_epochs = 10
print("!!!!! ****  in train 1/2  **** !!!!")
# try:
loss = []
acc = []
val_acc = []
val_loss = []

history = model.fit_generator(
    generator=train_generator,
    epochs=num_epochs,
    # steps_per_epoch=int(math.ceil(34564/float(self.config.trainer.batch_size))),
    # 34564 math.ceil(len(os.listdir(self.config.data_loader.train_dir))\
    #  / float(self.config.trainer.batch_size)), # WILL USE __len__() if left
    verbose=True,
    validation_data=test_generator,
    # validation_steps=int(math.ceil(8640/float(self.config.trainer.batch_size))),
    # math.ceil(len(os.listdir(self.config.data_loader.test_dir))\
    #  / float(self.config.trainer.batch_size)),
    callbacks=callbacks,
    use_multiprocessing=True,
    workers=12,#psutil.cpu_count(),
    # max_queue_size=20,
    shuffle=True
)
# except Exception as e:
    # print("!!! some problem happened with fit_generator()")
    # print(e)
print("!!!!! ****  in train 2/2  **** !!!!!")
loss.extend(history.history['loss'])
acc.extend(history.history['acc'])
val_loss.extend(history.history['val_loss'])
val_acc.extend(history.history['val_acc'])

