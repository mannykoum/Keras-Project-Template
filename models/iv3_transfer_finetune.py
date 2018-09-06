from base.base_model import BaseModel
from keras.models import Sequential, Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers \
import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D
from keras.optimizers import SGD, Adadelta
from keras.utils.vis_utils import plot_model
from keras.utils import multi_gpu_model
import os

#TODO: turn these into hyperparams
FC_SIZE = 1024

class IV3TransferModel(BaseModel):
    def __init__(self, config):
        super(IV3TransferModel, self).__init__(config)
        self.build_model()

    def setup_to_transfer_learn(self, base_model):
        """Freeze all layers and compile the model"""
        for layer in base_model.layers:
            layer.trainable = False
        # self.model.compile(optimizer=self.config.model.optimizer,
        #     loss=self.config.model.loss, metrics=['accuracy'])

    def add_new_last_layer(self, base_model, nb_classes):
        """Add last layer to the convnet
        Args:
        base_model: keras model excluding top
        nb_classes: # of classes
        Returns:
        new keras model with last layer
        """
        # TODO:OOP-ify
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
        predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

    def setup_to_finetune(self, nb_frozen_layers):
        """Freeze the bottom IV3 layers and retrain the remaining top layers.
        Args:
        nb_frozen_layers: how many of the bottom layers to freeze
        """
        for layer in self.model.layers[:int(nb_frozen_layers)]:
            layer.trainable = False
        for layer in self.model.layers[int(nb_frozen_layers):]:
            layer.trainable = True
        # self.model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),
        #         loss=self.config.model.loss, metrics=['accuracy'])

    def build_model(self):
        base_model = InceptionV3(input_shape=self.config.model.input_shape,
                weights='imagenet',
                include_top=False) #include_top=False excludes final FC layer
        self.model = self.add_new_last_layer(base_model, self.config.model.output_shape)


        # choose to finetune or transfer
        # TODO: clearly define the difference and make this more elegant
        if hasattr(self.config,"finetune"):
            if (self.config.model.finetune == True):
                nb_IV3_layers_to_freeze = self.config.model.frozen_layers
                self.setup_to_finetune(nb_IV3_layers_to_freeze)
            else:
                self.setup_to_transfer_learn(base_model)
        else:
            self.setup_to_transfer_learn(base_model)

        if (self.config.model.optimizer=="sgd"):
            opt = SGD(
                    lr=self.config.model.learning_rate,
                    momentum=self.config.model.momentum
            )
        elif (self.config.model.optimizer=="adadelta"):
            opt = Adadelta(
                    lr=self.config.model.learning_rate,
                    rho=self.config.model.rho
            )
        else:
            opt = self.config.model.optimizer

        # if the config has the save_plot flag is on, save plot
        # TODO: make dir robust, maybe add parent directory thingy
        if hasattr(self.config,"save_plot"):
            if (self.config.model.save_plot == True):
                plt_fname = 'models/plots/'+self.config.exp.name+'_model_plot.png'
                plot_model(self.model, to_file=plt_fname,
                        show_shapes=True, show_layer_names=True)
                if hasattr(self.config, "comet_api_key"):
                    self.config.exp_handle.set_model_graph(self.model.to_json())
                    #TODO: make sure this shit works
                    self.config.exp_handle.set_code(os.path.realpath(__file__))

        # On model saving
        # To save the multi-gpu model, use .save(fname) or .save_weights(fname)
        # with the template model (the argument you passed to multi_gpu_model),
        # rather than the model returned by multi_gpu_model.
        self.model = multi_gpu_model(self.model, gpus=2)

        self.model.compile(
              loss=self.config.model.loss,
              optimizer=opt,
              metrics=['accuracy'])

