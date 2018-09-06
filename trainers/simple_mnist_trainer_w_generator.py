
from base.base_trainer import BaseTrain
import sys, os, psutil, math
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import collections # for flatten() TODO: make a utility libr for flatten, etc.

def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class SimpleMnistModelTrainerWGenerator(BaseTrain):
    def __init__(self, model, train_generator, config, valid_data=None):
        super(SimpleMnistModelTrainerWGenerator, self).__init__(model,
                            train_generator, config, valid_data)
        # self.train_generator = train_generator
        # self.validation_generator = valid_data
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()



    def init_callbacks(self):
        # TODO: figure out LearningRateScheduler callback with polydecay maybe
        # TODO: early stopping callback
        self.callbacks.append(
            ModelCheckpoint(
                filepath=os.path.join(self.config.callbacks.checkpoint_dir,
                    '%s-{epoch:02d}-{val_loss:.2f}.hdf5' % self.config.exp.name),
                monitor=self.config.callbacks.checkpoint_monitor,
                mode=self.config.callbacks.checkpoint_mode,
                save_best_only=self.config.callbacks.checkpoint_save_best_only,
                save_weights_only=self.config.callbacks.checkpoint_save_weights_only,
                verbose=self.config.callbacks.checkpoint_verbose,
            )
        )

        self.callbacks.append(
            TensorBoard(
                log_dir=self.config.callbacks.tensorboard_log_dir,
                write_graph=self.config.callbacks.tensorboard_write_graph,
            )
        )

        self.callbacks.append(
            EarlyStopping(
                monitor=self.config.callbacks.early_stopping_monitor,
                min_delta=self.config.callbacks.early_stopping_min_delta,
                patience=self.config.callbacks.early_stopping_patience,
                verbose=self.config.callbacks.early_stopping_patience
            )
        )

        # if the config has the debug flag on, turn on tfdbg (TODO: make it work)
        if hasattr(self.config,"debug"):
            if (self.config.debug == True):
                import keras.backend
                from tensorflow.python import debug as tf_debug
                print("#=========== DEBUG MODE ===========#")
                sess = keras.backend.get_session()
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                keras.backend.set_session(sess)

        # if the config file has a comet_ml key, log on comet
        if hasattr(self.config,"comet_api_key"):
            #from comet_ml import Experiment # PUT the import in main
            #experiment = Experiment(api_key=self.config.exp.comet_api_key,
            #    project_name=self.config.exp.name)
            #experiment.disable_mp()
            self.config.exp_handle.log_multiple_params(flatten(self.config.toDict()))
            self.callbacks.append(self.config.exp_handle.get_keras_callback())

    def train(self):

        # TODO: fix this it's now here just for future sanity
        # (somehow split the incoming data?)
        if (self.valid_data==None):
            print("""Need some validation data or the
                ModelCheckpoint won't be saved""")
            return

        history = self.model.fit_generator(
            generator=self.data,
            epochs=self.config.trainer.num_epochs,
            # steps_per_epoch=int(math.ceil(34564/float(self.config.trainer.batch_size))),
            # 34564 math.ceil(len(os.listdir(self.config.data_loader.train_dir))\
            #  / float(self.config.trainer.batch_size)), # WILL USE __len__() if left
            verbose=self.config.trainer.verbose_training,
            validation_data=self.valid_data,
            # validation_steps=int(math.ceil(8640/float(self.config.trainer.batch_size))),
            # math.ceil(len(os.listdir(self.config.data_loader.test_dir))\
            #  / float(self.config.trainer.batch_size)),
            callbacks=self.callbacks,
            use_multiprocessing=True,
            workers=psutil.cpu_count(),
            # max_queue_size=20,
            shuffle=True
        )

        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
