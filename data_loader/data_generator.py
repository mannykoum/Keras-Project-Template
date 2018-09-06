import numpy as np
from skimage import transform, io
from sklearn import preprocessing # for label_binarizer
from keras import utils # for Sequence, to_categorical

class DataGenerator(utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, data_dir, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True, class_mode=None):
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
        if class_mode:
            self.encode_fcn = self.loss2encode(class_mode)
        else: # default is categorical_crossentropy
            self.encode_fcn = utils.to_categorical
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.list_IDs) / float(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.ctr = 0
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels)) # Python 3
        # X = np.empty(self.tuple_wrap(self.batch_size, self.dim, self.n_channels)) # Python 2
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            # X[i,] = np.load('data/' + ID + '.npy')
            X_tmp = np.array(io.imread(self.data_dir+ID+'.png', as_gray=True))
            X[i,] = X_tmp.reshape(*self.dim, self.n_channels) # Python 3
            # X[i,] = X_tmp.reshape(self.tuple_follow(self.dim, self.n_channels)) # Python 2

            # Store class directly from label dictionary
            y[i] = self.labels[ID]

        return X, self.encode_fcn(y, num_classes=self.n_classes)

    def loss2encode(self, loss):
        # TODO: add more
        losses = {
            "sparse_categorical_crossentropy": self.to_sparse,
            "categorical_crossentropy": utils.to_categorical,
            "binary_crossentropy": self.to_binary,
        }
        return losses[loss]

    def to_sparse(self, y, num_classes=0):
        # TODO: make this robust
        for el in y:
            el = int(el)
        return y

    def to_binary(self, y, num_classes=0):
        # TODO: make this robust; now only works with numerical labels
        classes = np.arange(num_classes)
        return preprocessing.label_binarize(y, classes)

    def data_summary(self, X, y):
        """Summarize current state of dataset"""
        print('images shape:', X.shape)
        print('labels shape:', y.shape)
        print('images shape:', type(X))
        print('labels shape:', type(y))
        # print('labels:', y)

    def tuple_wrap(self, val1, tupl, val2):
        'a wrapping function because unpacking doesn\'t work in Python-2.x'
        lst = [val1]
        for el in tupl:
            lst.append(el)
        lst.append(val2)
        return tuple(lst)

    def tuple_follow(self, tupl, val):
        '''a function that adds a val to a tuple because 
        unpacking doesn\'t work in Python-2.x'''
        for el in tupl:
            lst.append(el)
        lst.append(val)
        return tuple(lst)