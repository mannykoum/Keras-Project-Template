import numpy as np
from skimage import transform, io
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, data_dir, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.data_dir = data_dir
        if not(callable(labels)):
            self.labels = labels
            # print("LABELS ARE "+str(labels))
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.ctr = 0
        print("initialized")
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        print("length is : ", int(np.ceil(len(self.list_IDs) / float(self.batch_size))))
        return int(np.ceil(len(self.list_IDs) / float(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        print('in get_item 1/2')
        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        # print(X)
        # print(Y)
        # print("y LABELS have shape " + str(y.shape))
        # print("get item called ", self.ctr, " times")
        # self.data_summary(X, y)
        # self.ctr+=1
        # print(self.ctr, " steps up to this batch")
        print('in get_item 2/2')
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        print("___----==== on epoch end called ====----____")
        print(self.ctr, " steps this epoch")
        self.ctr = 0
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)
        print('in data_gen')
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            print('enumerating')
            X_tmp = np.array(io.imread(self.data_dir+ID+'.png', as_gray=True))
            X[i,] = X_tmp.reshape(*self.dim, self.n_channels)

            # Store class directly from label dictionary
            y[i] = self.labels[ID]

        # print(keras.utils.to_categorical(y, num_classes=self.n_classes))
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


    def data_summary(self, X, y):
        """Summarize current state of dataset"""
        print('images shape:', X.shape)
        print('labels shape:', y.shape)
        print('images shape:', type(X))
        print('labels shape:', type(y))
        # print('labels:', y)
