from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
from utils import factory
import sys

 
def main():
    # capture the config path from the run arguments
    # then process the json configuration fill
#    try:
    # Parse args
    args = get_args()
    config = process_config(args.config)

    # if debug mode is on, add it on the config dotmap
    if args.debug == True:
        from dotmap import DotMap
        config.debug = True

    # comet_ml needs to be imported before Keras
    if hasattr(config,"comet_api_key"):
        from comet_ml import Experiment
    
    ## extra imports to set GPU options
    import tensorflow as tf
    from keras import backend as k
    ###################################
    # TensorFlow wizardry
    tf_config = tf.ConfigProto()
     
    # Don't pre-allocate memory; allocate as-needed
    tf_config.gpu_options.allow_growth = True
     
    # Only allow a total of half the GPU memory to be allocated
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.9
     
    # Create a session with the above options specified.
    k.tensorflow_backend.set_session(tf.Session(config=tf_config))


    # create the experiments dirs
    create_dirs([config.callbacks.tensorboard_log_dir, config.callbacks.checkpoint_dir])

    print('Create the data generator.')
    data_loader = factory.create("data_loader."+config.data_loader.name)(config)

    print('Create the model.')
    model = factory.create("models."+config.model.name)(config)

    print('Create the trainer')
    trainer = factory.create("trainers."+config.trainer.name)(model.model, data_loader.get_train_data(), config, validation_generator=data_loader.get_test_data())

    print('Start training the model.')
    trainer.train()

#    except Exception as e:
#        print(e)
#        sys.exit(1)

if __name__ == '__main__':
    main()
