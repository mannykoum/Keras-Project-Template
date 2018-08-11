from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils.vis_utils import plot_model
from keras.utils.training_utils import multi_gpu_model


class GEO1dConvModel(BaseModel):
    def __init__(self, config):
        super(GEO1dConvModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu', 
                         input_shape=self.config.model.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.config.model.output_shape, 
                         activation='sigmoid'))

                # if the config has the debug flag on, turn on tfdbg (TODO: make it work)
        if hasattr(self.config,"debug"):
            if (self.config.debug == True):
                print(self.model.summary())
                plot_model(self.model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        # TODO: check num_gpus
        self.model = multi_gpu_model(self.model, gpus=2)

        self.model.compile(
              loss='sparse_categorical_crossentropy',
              optimizer=self.config.model.optimizer,
              metrics=['accuracy'])