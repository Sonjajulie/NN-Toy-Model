
# cnn model
# https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/
import numpy as np
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.callbacks import ModelCheckpoint
class CNN1D:
    """ Train and execute CNN model"""
    def __init__(self,standardize_bool,data):
        if standardize_bool:
            self.calculate_standardized_values(data)

    def calculate_standardized_values(self,data):
        # # For StandardScaler the data needs to be of the shape: [number_of_samples, number_of_features]
        # # => sce and sic should be independend!
        # #  2
        # # [number_of_samples, number_of_features]
        # scaler = StandardScaler()
        # data_standardized = [scaler.fit_transform(data[i]) for i in range(len(data))]
        # return data_standardized
        # values 2
        self.data_mean = np.mean(data, axis=1)
        # std 2
        self.sigma_var = np.std(data, axis=1)
        self.data_standardized = [(data[i] - self.data_mean[i]) / self.sigma_var[i] for i in range(len(self.sigma_var))]
        return self.data_standardized


    def evaluate_model(self,trainX, trainy, testX, testy,X_val,y_val, n_kernel):
        """ fit and evaluate a model """
        verbose, epochs, batch_size = 0, 100, 64
        # first need to reshape data X according to
        # https: // stackoverflow.com / questions / 43396572 / dimension - of - shape - in -conv1d / 43399308
        X_train = np.expand_dims(trainX, axis=2)  # reshape (lenX,lenY, 1)
        X_test = np.expand_dims(testX, axis=2)  # reshape (lenX,lenY, 1)
        X_val = np.expand_dims(X_val, axis=2)  # reshape (lenX,lenY, 1)
        n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], trainy.shape[1]
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=n_kernel, activation='relu', input_shape=(n_timesteps,n_features)))
        model.add(Conv1D(filters=32, kernel_size=n_kernel, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(200, activation='relu'))
        model.add(Dense(n_outputs, activation='linear'))
        # model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
        model.compile( loss='mean_squared_error' , optimizer='adam' , metrics=['mean_squared_error'])
        # fit network
        filepath = f"ModelWeights.hdf5"
        checkpoint = ModelCheckpoint(filepath, save_best_only=True, monitor="val_mean_squared_error")


        out=model.fit(X_train, trainy, epochs=epochs, batch_size=batch_size, validation_data=(X_val,y_val), verbose=verbose, callbacks=[checkpoint])
        model.save(f'dnn_model.h5')
        if 0:
            plt.figure(1, figsize=(12, 6))
            plt.clf()
            plt.semilogy(out.history["loss"][:])
            plt.legend(['log(loss)'])
            plt.xlabel('epoch')
            plt.ylabel('$\log_{10}$(loss)')
            plt.savefig(f"Progress_of_Optimization_cnn1D.pdf", bbox_inches='tight')
            plt.pause(0.05)

            plt.figure(2, figsize=(12, 6));
            plt.clf()
            plt.plot(model.history.history['mean_squared_error'])
            plt.plot(model.history.history['val_mean_squared_error'])
            plt.title("Model's Training & Validation loss across epochs")
            plt.ylabel('Loss')
            plt.xlabel('Epochs')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.savefig(f"Loss_function_train_and_loss_cnn1D.pdf", bbox_inches='tight')

            plt.figure(3, figsize=(12, 6))
            plt.clf()
            plt.plot(model.history.history['mean_squared_error'])
            plt.plot(model.history.history['val_mean_squared_error'])
            plt.title("Model's Training & Validation mean squared error across epochs")
            plt.ylabel('mean squared error')
            plt.xlabel('Epochs')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.savefig(f"mean_squared_error_train_and_loss_cnn1D.pdf", bbox_inches='tight')

        print('Testing the model performance...')
        # evaluate model
        _, mean_squared_error = model.evaluate(X_test, testy, batch_size=batch_size, verbose=0)
        ##### Testing the model performance

        print(X_test.shape)
        y1 = model.predict(X_test)
        # print(f"y1=model.predict(X_test_point)={y1}");
        # print(f"y_real={prcp.get_data_point_from_precursors(X_test_point[0])}");
        # Use the model's evaluate method to predict and evaluate the test datasets
        # Print the results
        print(y1.shape)
        result = model.evaluate(X_test, testy)
        for i in range(len(model.metrics_names)):
            print(f"Metric {model.metrics_names[i]}: {round(result[i], 5)}")
        return mean_squared_error,y1

    # summarize scores
    def summarize_results(self,scores, params):
        print(scores, params)
        # summarize mean and standard deviation
        for i in range(len(scores)):
            m, s = mean(scores[i]), std(scores[i])
            print('Param=%d: %.3f%% (+/-%.3f)' % (params[i], m, s))
        # boxplot of scores
        pyplot.boxplot(scores, labels=params)
        pyplot.savefig('exp_cnn_kernel.png')

    # run an experiment
    def run_experiment(self,trainX, trainy, testX, testy, X_val, y_val, params,repeats=10):
        # score,y1 = self.evaluate_model(trainX, trainy, testX, testy, X_val, y_val, 3)

        # test each parameter
        all_scores = list()
        for p in params:
            # repeat experiment
            scores = list()
            for r in range(repeats):
                score,y1 = self.evaluate_model(trainX, trainy, testX, testy,X_val,y_val,p)
                score = score * 100.0
                print('>p=%d #%d: %.3f' % (p, r + 1, score))
                scores.append(score)
            all_scores.append(scores)
        # summarize results
        self.summarize_results(all_scores, params)
        return y1