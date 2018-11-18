from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
import pandas as pd

class Dataset:

    def __init__(self, train, test):
        self._train = train
        self._test = test

    def get_generators(self, length, sampling_rate, stride, batch_size):
        '''
        This function creates generators to be used with the model.fit_generator
        method of keras's Models. The data is normalized to be between 0 and 1.

        Parameters
        --------------------------------
        For explanation of input parameters consult keras documentation:
        https://keras.io/preprocessing/sequence/

        Returns
        --------------------------------
        dict
            Returns a dictionary of generators whose keys are the stocks
            that the generator corresponds to.
        '''
        generators = {}
        scaler = MinMaxScaler(feature_range=(0, 1))
        for stock in list(self._train.columns.unique(level=0)):
            data = scaler.fit_transform(self._train[stock])
            generators[stock] = TimeSeriesGenerator(data, data, length=length, 
                                                    sampling_rate=sampling_rate, stride=stride, batch_size=batch_size)

        return generators

    @staticmethod
    def read(path, test_length, stocks=[], interpolation_method='linear'):
        '''
        This function splits the dataset into train and test sets based on the desired
        length of time of the test set in minutes.

        Parameters
        ------------------------------
        path : string
            Path to the dataset csv file
        test_length : int
            Length in minutes of the test set
        stocks : list
            List of desired stocks in train and test sets
        interpolation_method : string
            The method used to interpolate nan values. Consult
            pandas documentation for options:
            https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.interpolate.html

        Returns
        --------------------------------
        Dataset
            Dataset object for manipulating the data, mainly for calling
            the generator generation method.
        '''
        if not stocks:
            raise ValueError

        dataset = pd.read_csv(path, index_col=0, header=[0, 1]).sort_index(axis=1)
        train_end = dataset.shape[0] - test_length
        dataset = dataset[stocks]

        for stock in stocks:
            dataset[stock] = dataset[stock].interpolate(method=interpolation_method)

        return Dataset(dataset.iloc[0:train_end], dataset.iloc[train_end::])
