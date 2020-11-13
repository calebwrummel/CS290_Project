import numpy as np
import pandas as pd
import keras
from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import models, layers, Input, regularizers, losses, metrics, optimizers
from tensorflow.keras.models import Sequential, Model, load_model
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn

data = pd.read_csv("data.csv") # Importing the data
data.drop(data[data.year <1980].index, inplace = True) # Dropping classic cars
data.drop(data[data.con_int == 1].index, inplace = True) # Dropping cars that are being sold as scrap
data.drop(data[data.con_int == 6].index, inplace = True) # Dropping new cars
data.drop(data[data.price < 1000].index, inplace = True) # Dropping cars with super low prices, most likely erroneous
data.drop(data[data.price > 75000].index, inplace = True) # Cutting off expensive cars
data = data.drop(columns=['Unnamed: 0', 'region', 'manufacturer', 'model', 'condition', 'transmission', 'model_int'])
#replacing missing mileage values w average 
miles_avg = np.mean(data["odometer"])
data['odometer'] = data['odometer'].fillna(miles_avg)
#replace missing year w avg year
year_avg = np.mean(data["year"])
data['year'] = data['year'].fillna(year_avg)
y = data['price']
x = data.drop(columns=['price'])

x = x.to_numpy()
y = y.to_numpy()
scaler = StandardScaler() # Neural networks play best with numbers between 0 and 1 
# so we standardize the data (make it have a Guassian distribution with mean 0 and unit variance)
x = scaler.fit_transform(x)
x_train, x_val, og_y_train, og_y_val = train_test_split(x, y, test_size=0.2, random_state=2, shuffle=True)
y_train = scaler.fit_transform(np.reshape(og_y_train,(len(og_y_train),1)))
y_val = scaler.fit_transform(np.reshape(og_y_val,(len(og_y_val),1)))

def log_dir_name(num_dense_nodes1,
                 num_dense_nodes2):
    
    """
    Create the name for each model which will be saved along with its performance in a .txt file.
    Named for the hyperparameter we are optimizing - the number of nodes in each hidden layer.
    """

    s = "{0} {1}"

    log_dir = s.format(num_dense_nodes1,
                       num_dense_nodes2)

    return log_dir

def create_model(num_dense_nodes1, 
                 num_dense_nodes2):
    
    """
    Creates a model based on certain values for the hyperparameters we are optimizing.
    """
    
    model = keras.Sequential()
    model.add(layers.Dense(num_dense_nodes1, activation="relu", input_shape=(6,)))
    model.add(layers.Dense(num_dense_nodes2, activation="relu"))
    model.add(layers.Dense(1, activation = None))
    model.compile(loss='mae', optimizer= 'adam')
    
    return model

"""
Performing grid search on our hyperparameters. Instead of trying every single number of nodes between 1 and a 100 
which would take a ton of time since we don't have great access to GPU researches, we're only taking a look at
every possible number of nodes between 1 and 100 in increments of 10. Only have 6 input features so figured we'd cap
the input space at 100.
"""
dim1 = np.arange(61,100,10)
dim2 = np.arange(1,100,10)
for i in dim1:
    for j in dim2:
        K.clear_session()
        model = create_model(i,j)
        
        """
        Use the early stopping callback to prevent overfitting. This stops training after the model 
        doesn't see an increase in performance (in real terms = a decrease in loss) on validation
        data after 3 epochs, and when that happens it reverts the model's weights to the best ones before the performance stopped dropping.
        """
        
        callback_log = [
        keras.callbacks.EarlyStopping(monitor='val_loss',
                                      patience=3,
                                      restore_best_weights=True)]
        history = model.fit(x_train,
                        y_train,  
                        validation_data=(x_val,
                                         y_val), 
                        batch_size=128, 
                        epochs=500, 
                        callbacks=callback_log)
    

        y_pred = model.predict(x_val)
        y_pred = scaler.inverse_transform(y_pred) # To see how the model is doing in real terms we un-scale the data then compute the MAE.
        mae = sklearn.metrics.mean_absolute_error(og_y_val, y_pred)
        # Print the MAE
        print(i, j, mae)
        log_dir = log_dir_name(i,j)

    # Save the name of the model, its performance, and the number of epochs it trained for in a .txt file.
        with open("scores.txt", "a") as f:
            f.write("{} {} {}\n".format(log_dir, mae, len(history.history['loss'])))