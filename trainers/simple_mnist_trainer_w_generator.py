
from base.base_trainer import BaseTrain
import os, psutil
from keras.callbacks import ModelCheckpoint, TensorBoard

class SimpleMnistModelTrainerWGenerator(BaseTrain):
    def __init__(self, model, train_generator, config, validation_generator=None):
        super(SimpleMnistModelTrainerWGenerator, self).__init__(model, 
                                                        train_generator, config)
        self.train_generator = train_generator
        self.validation_generator = validation_generator
        self.callbacks = []
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []
        self.init_callbacks()

    def init_callbacks(self):
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
            from comet_ml import Experiment # PUT the import in main
            experiment = Experiment(api_key=self.config.exp.comet_api_key, 
                project_name=self.config.exp.name)
            experiment.disable_mp()
            experiment.log_multiple_params(self.config)
            self.callbacks.append(experiment.get_keras_callback())

    def train(self):
        print("!!!!! ****  in train 1/2  **** !!!!")
        history = self.model.fit_generator(
            generator=self.train_generator,
            epochs=self.config.trainer.num_epochs,
            # steps_per_epoch=(len(os.listdir(self.config.data_loader.train_dir))\
            #  / self.config.trainer.batch_size), WILL USE __len__()
            verbose=self.config.trainer.verbose_training,
            validation_data=self.validation_generator,
            callbacks=self.callbacks,
            use_multiprocessing=False,
            workers=12,#psutil.cpu_count(),
            # max_queue_size=20,
            shuffle=True
        )
        print("!!!!! ****  in train 2/2  **** !!!!")
        self.loss.extend(history.history['loss'])
        self.acc.extend(history.history['acc'])
        self.val_loss.extend(history.history['val_loss'])
        self.val_acc.extend(history.history['val_acc'])
