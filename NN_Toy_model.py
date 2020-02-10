## Regression of 2d data using Neural networks in Python
## Eli Tziperman, APM120, 201709
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import matplotlib

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.datasets import imdb
from MixtureModel import MixtureGaussianModel
from Predictand import Predictand

if __name__=='__main__':

    # Create gaussian distributions for precursors
    # first 2 points: location snow cover extent (sce)
    # last 2 points:  location sea ice concentration (sic)
    # mean == composites, sigma == noise
    gaussian_distributions = [
    {"mean": [-1, 1, 1, -1], "sigma": [[0.01,0.,0.,0.], [0.,0.01,0.,0.],[0.,0.,0.01,0.],[0.,0.,0.,0.01]]},
    {"mean": [-1, 0, 1, 1] , "sigma": [[0.01,0.,0.,0.], [0.,0.01,0.,0.],[0.,0.,0.01,0.],[0.,0.,0.,0.01]]},]
    
    # create time series
    t_end=5000
    time_series=range(t_end)
    
    # create instance to get samples for sic and sce
    precursors = MixtureGaussianModel(gaussian_distributions)
    # get samples
    X = (precursors.rvs(t_end))
    
    # array which lead with composites to clusters pf PRCP
    array = np.array([[1,2,1,1],[-0.5,0,-0.5,1.],[-1,0,-1,-1]], np.float)
    prcp_clusters = [{"cluster": [-1, 1, 1, -1]},{"cluster": [-1, 1, 1, -1]}]
    prcp = Predictand(prcp_clusters,array)
    y = prcp.get_data_from_precursors(X)
    
    # to optimize, need to adjust learning rate (lr) and its decay, and batch_size.
#
    print('Create network (model): specify number of neurons in each layer:')
    nr_neurons = 8
    np.random.seed(3456);
    model = Sequential()#kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None), bias_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.02, seed=None)

    model.add(Dense(nr_neurons, input_dim=4, activation='relu',
kernel_regularizer=regularizers.l2(0.0001)#sigmoid
))
    model.add(Dense(nr_neurons,  activation='relu',
kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Dense(nr_neurons,  activation='relu',
kernel_regularizer=regularizers.l2(0.0001)
))

    model.add(Dense(3, activation='linear'))

    print('Compile model')

    model.compile( 
#        loss='categorical_crossentropy'
        loss='mean_squared_error'
        ,optimizer=optimizers.Adam()#lr=0.2,decay=0.001
        ,metrics=['mean_squared_error'])# mean_squared_error

#    model.compile(optimizer = "Adam",loss="binary_crossentropy",
#    metrics=["accuracy"])mean_squared_error




    from sklearn.preprocessing import StandardScaler
    if 1:
        yT=y.T
        XT=X.T
        print('normalize inputs')
        Xmean=np.mean(XT); Xstd=np.std((XT-Xmean).T).T; 
        ymean=np.mean(yT); ystd=np.std((yT-ymean).T).T; 
        X=(XT-Xmean)/Xstd;
        y=(yT-ymean)/ystd;
    else:
        Xstd=1; ystd=1; Xmean=0; ymean=0;

####  Create training and test data set for NN
    from sklearn.model_selection import train_test_split
    #Create train and test dataset with an 80:20 split
    X_train, X_test, y_train, y_test = train_test_split(X.T, y.T,test_size=0.2,random_state=2019)
    #Further divide training dataset into train and validation dataset with an 90:10 split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=0.1,random_state=2019)

#### Check the sizes of all newly created datasets
    print("Shape of x_train:",X_train.shape)
    print("Shape of x_val:",X_val.shape)
    print("Shape of x_test:",X_test.shape)
    print("Shape of y_train:",y_train.shape)
    print("Shape of y_val:",y_val.shape)
    print("Shape of y_test:",y_test.shape)


    print('Train (fit) the network...')
#    mcp = ModelCheckpoint("Output/regression_checkpoint.dat"
#                          , monitor="mean_squared_error"
#                          , save_weights_only=False)
    filepath = f"ModelWeights-{nr_neurons}.hdf5"
    checkpoint = ModelCheckpoint(filepath, save_best_only=True,
    monitor="val_mean_squared_error")
    out=model.fit(X_train, y_train
                  , validation_data=(X_val,y_val),
                  epochs=500, batch_size=32, verbose=0, callbacks=[checkpoint]) # 
                  # , epochs=500, batch_size=30, verbose=0, callbacks=[mcp])

    # Saves the entire model into a file named as  'dnn_model.h5'
    model.save(f'dnn_model-{nr_neurons}.h5')

    print('initial loss='+repr(out.history["loss"][1])
          +', final='+repr(out.history["loss"][-1]))
    print(repr(out))
#### plot progress of optimization
if 1:
    plt.figure(1,figsize=(12,6)); plt.clf();
    plt.semilogy(out.history["loss"][:])
    plt.legend(['log(loss)'])
    plt.xlabel('epoch');
    plt.ylabel('$\log_{10}$(loss)');
    plt.savefig(f"Progress_of_Optimization_{nr_neurons}_neurons.pdf", bbox_inches='tight')
    plt.pause(0.05)
    
    plt.figure(2,figsize=(12,6)); plt.clf();
    plt.plot(model.history.history['mean_squared_error'])
    plt.plot(model.history.history['val_mean_squared_error'])
    plt.title("Model's Training & Validation loss across epochs")
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f"Loss_function_train_and_loss_{nr_neurons}_neurons.pdf", bbox_inches='tight')
    
    plt.figure(3,figsize=(12,6)); plt.clf();
    plt.plot(model.history.history['mean_squared_error'])
    plt.plot(model.history.history['val_mean_squared_error'])
    plt.title("Model's Training & Validation mean squared error across epochs")
    plt.ylabel('mean squared error')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(f"mean_squared_error_train_and_loss_{nr_neurons}_neurons.pdf", bbox_inches='tight')




#### Use the trained network to classify a new point:
X_test_point = np.array([[-1.009,0.193,0.940,1.058]])

y1 = model.predict((X_test_point-Xmean)/Xstd)*ystd+ymean;
print(f"XtestPoint= {X_test_point}");
print(f"y1=model.predict(X_test_point)={y1}");
print(f"y_real={prcp.get_data_point_from_precursors(X_test_point[0])}");



##### Testing the model performance
print('Testing the model performance...')
#Use the model's evaluate method to predict and evaluate the test datasets
result = model.evaluate(X_test,y_test)
#Print the results
for i in range(len(model.metrics_names)):
    print(f"Metric {model.metrics_names[i]}: {round(result[i],5)}")
        

#### Plot data points
from PlotDataPoints import PlotDataPoints
#    
result_list=[]
n = len(X_test)
result_list=(model.predict(X_test))
pdp=PlotDataPoints(f"Performance_of_NN_Test_{nr_neurons}_Neurons.pdf")
time=[i for i in range(n)]
pdp.plot_data_true_modelled(time,(y_test.T),(result_list.T))
#pdp.plot_data(time,X_test.T,f"sce_sic_{nr_neurons}_Neurons.pdf")