## Regression of 2d data using Neural networks in Python
## Eli Tziperman, APM120, 201709
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.datasets import imdb


from tensorflow.keras.layers import Conv1D
from MixtureModel import MixtureGaussianModel
from Predictand import Predictand
from PlotDataPoints import PlotDataPoints
from Cnn1D import CNN1D
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split



if __name__=='__main__':

    # Create gaussian distributions for precursors
    # first 2 points: location snow cover extent (sce)
    # last 2 points:  location sea ice concentration (sic)
    # mean == composites, sigma == noise
    gaussian_distributions = [
    {"mean": [-1, 1, 1, -1], "sigma": [[0.01,0.,0.,0.], [0.,0.01,0.,0.],[0.,0.,0.01,0.],[0.,0.,0.,0.01]]},
    {"mean": [-1, 0, 1, 1] , "sigma": [[0.01,0.,0.,0.], [0.,0.01,0.,0.],[0.,0.,0.01,0.],[0.,0.,0.,0.01]]},]
    
    # create time series
    t_end=2000
    time_series=range(t_end)
    
    # create instance to get samples for sic and sce
    precursors = MixtureGaussianModel(gaussian_distributions)
    # get samples
    sample_spatial=1000

    model_dict = (precursors.rvs_2d(t_end,sample_spatial))
    # needs to be of the form [samples, time steps, features] so in my case 2 X 500 for CNN1D
    # Try to use only one feature
    # X = model_dict['X_feature']
    X = model_dict['X']
    print("Test")
    print(model_dict['X'].shape,X.shape,np.max(X),np.min(X))

#     # array which lead with composites to clusters pf PRCP
    array = np.array([[1,2,1,1],[-0.5,0,-0.5,1.],[-1,0,-1,-1]], np.float)
    prcp_clusters = [{"cluster": [1,-1, 1]},{"cluster": [1, 1, -1]}]
    prcp = Predictand(prcp_clusters,array)
    len_precip: int = 100

    prcp_data = prcp.get_data_from_precursors_2d(model_dict['mean'],model_dict['std'],t_end,len_precip)
    y = prcp_data["X"]
    print(y.shape)
    pdp = PlotDataPoints(f"Performance_of_NN_Test_Neurons.pdf")
    print('Create network (model): specify number of neurons in each layer:')
    pdp.array_plot(t_end,sample_spatial,len_precip,model_dict['X'],y,"data.png")
    # # Normalize input and output data
    cnn1d = CNN1D(True,X)
    # #Create train and test dataset with an 80:20 split

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=2019)
    print("Train")
    print(X_train.shape,X_test.shape,np.max(X),np.min(X))

    # #Further divide training dataset into train and validation dataset with an 90:10 split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.1,random_state=2019)
    print("Vale")
    print(X_train.shape,X_val.shape,np.max(X),np.min(X))

    n_params = [2, 3, 5, 7, 11]
    y1 = cnn1d.run_experiment(X_train, y_train, X_test, y_test,X_val,y_val, n_params,repeats=10)
    pdp.array_plot2(len(X_test), sample_spatial, len_precip, X_test, y1, "data_CNN1.png")

    pdp.array_plot2(len(X_test), sample_spatial, len_precip, X_test, y_test, "data_True.png")
    # pdp.array_plot(t_end,sample_spatial,len_precip,model_dict['X'],y,"data.png")

    #### Plot data points
    # from PlotDataPoints import PlotDataPoints
    #
    # #
    # result_list = []
    # n = len(X_test)
    # result_list = (model.predict(X_test))
    # pdp = PlotDataPoints(f"Performance_of_NN_Test_{nr_neurons}_Neurons.pdf")
    # time = [i for i in range(n)]
    # pdp.plot_data_true_modelled(time, (y_test.T), (result_list.T))
    # pdp.plot_data(time,X_test.T,f"sce_sic_{nr_neurons}_Neurons.pdf")


    # y1 = model.predict((X_test_point - Xmean) / Xstd) * ystd + ymean;

    # nr_neurons = 1028
    # np.random.seed(3456);
    # model = Sequential()
    # model.add(Conv1D(filters=64,  kernel_size=3, activation='relu',input_shape=(model_dict['X'].shape[0],model_dict['X'].shape[1])))
    # model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(MaxPooling1D(pool_size=2))
    # model.add(Flatten())
    # model.add(Dense(100, activation='relu'))
    # model.add(Dense(len(y), activation='softmax'))
    # model.compile(loss='linear', optimizer='adam', metrics=['mean_squared_error']
    # print(model.summary())





#     model = Sequential()#kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)
#
#     model.add(Dense(nr_neurons, input_dim=4, activation='relu',
# kernel_regularizer=regularizers.l2(0.0001)#sigmoid
# ))
#     model.add(Dense(nr_neurons,  activation='relu',
# kernel_regularizer=regularizers.l2(0.0001)))
#     model.add(Dense(nr_neurons,  activation='relu',
# kernel_regularizer=regularizers.l2(0.0001)
# ))
#
#     model.add(Dense(3, activation='linear'))
#
#     print('Compile model')
#
#     model.compile(
# #        loss='categorical_crossentropy'
#         loss='mean_squared_error'
#         ,optimizer=keras.optimizers.Adam()#lr=0.2,decay=0.001
#         ,metrics=['mean_squared_error'])# mean_squared_error
#
# #    model.compile(optimizer = "Adam",loss="binary_crossentropy",
# #    metrics=["accuracy"])mean_squared_error
#
#
#
#
#     from sklearn.preprocessing import StandardScaler

#
# ####  Create training and test data set for NN
#     from sklearn.model_selection import train_test_split
#     #Create train and test dataset with an 80:20 split
#     X_train, X_test, y_train, y_test = train_test_split(X.T, y.T,test_size=0.2,random_state=2019)
#     #Further divide training dataset into train and validation dataset with an 90:10 split
#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
#                                                       test_size=0.1,random_state=2019)
#
# #### Check the sizes of all newly created datasets
#     print("Shape of x_train:",X_train.shape)
#     print("Shape of x_val:",X_val.shape)
#     print("Shape of x_test:",X_test.shape)
#     print("Shape of y_train:",y_train.shape)
#     print("Shape of y_val:",y_val.shape)
#     print("Shape of y_test:",y_test.shape)
#
#
#     print('Train (fit) the network...')
# #    mcp = ModelCheckpoint("Output/regression_checkpoint.dat"
# #                          , monitor="mean_squared_error"
# #                          , save_weights_only=False)
#     filepath = f"ModelWeights-{nr_neurons}.hdf5"
#     checkpoint = ModelCheckpoint(filepath, save_best_only=True,
#     monitor="val_mean_squared_error")
#     out=model.fit(X_train, y_train
#                   , validation_data=(X_val,y_val),
#                   epochs=500, batch_size=32, verbose=0, callbacks=[checkpoint]) #
#                   # , epochs=500, batch_size=30, verbose=0, callbacks=[mcp])
#
#     # Saves the entire model into a file named as  'dnn_model.h5'
#     model.save(f'dnn_model-{nr_neurons}.h5')
#
#     print('initial loss='+repr(out.history["loss"][1])
#           +', final='+repr(out.history["loss"][-1]))
#     print(repr(out))
# #### plot progress of optimization
# if 1:
#     plt.figure(1,figsize=(12,6)); plt.clf();
#     plt.semilogy(out.history["loss"][:])
#     plt.legend(['log(loss)'])
#     plt.xlabel('epoch');
#     plt.ylabel('$\log_{10}$(loss)');
#     plt.savefig(f"Progress_of_Optimization_{nr_neurons}_neurons.pdf", bbox_inches='tight')
#     plt.pause(0.05)
#
#     plt.figure(2,figsize=(12,6)); plt.clf();
#     plt.plot(model.history.history['mean_squared_error'])
#     plt.plot(model.history.history['val_mean_squared_error'])
#     plt.title("Model's Training & Validation loss across epochs")
#     plt.ylabel('Loss')
#     plt.xlabel('Epochs')
#     plt.legend(['Train', 'Validation'], loc='upper right')
#     plt.savefig(f"Loss_function_train_and_loss_{nr_neurons}_neurons.pdf", bbox_inches='tight')
#
#     plt.figure(3,figsize=(12,6)); plt.clf();
#     plt.plot(model.history.history['mean_squared_error'])
#     plt.plot(model.history.history['val_mean_squared_error'])
#     plt.title("Model's Training & Validation mean squared error across epochs")
#     plt.ylabel('mean squared error')
#     plt.xlabel('Epochs')
#     plt.legend(['Train', 'Validation'], loc='upper right')
#     plt.savefig(f"mean_squared_error_train_and_loss_{nr_neurons}_neurons.pdf", bbox_inches='tight')
#
#
#
#
# #### Use the trained network to classify a new point:
# X_test_point = np.array([[-1.009,0.193,0.940,1.058]])
#
# y1 = model.predict((X_test_point-Xmean)/Xstd)*ystd+ymean;
# print(f"XtestPoint= {X_test_point}");
# print(f"y1=model.predict(X_test_point)={y1}");
# print(f"y_real={prcp.get_data_point_from_precursors(X_test_point[0])}");
#
#
#
# ##### Testing the model performance
# print('Testing the model performance...')
# #Use the model's evaluate method to predict and evaluate the test datasets
# result = model.evaluate(X_test,y_test)
# #Print the results
# for i in range(len(model.metrics_names)):
#     print(f"Metric {model.metrics_names[i]}: {round(result[i],5)}")
#
#
# #### Plot data points
# from PlotDataPoints import PlotDataPoints
# #
# result_list=[]
# n = len(X_test)
# result_list=(model.predict(X_test))
# pdp=PlotDataPoints(f"Performance_of_NN_Test_{nr_neurons}_Neurons.pdf")
# time=[i for i in range(n)]
# pdp.plot_data_true_modelled(time,(y_test.T),(result_list.T))
# #pdp.plot_data(time,X_test.T,f"sce_sic_{nr_neurons}_Neurons.pdf")