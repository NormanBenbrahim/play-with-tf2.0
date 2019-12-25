from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np 
import pandas as pd 
import seaborn as sns # makes prettier plots than matplotlib
import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)

# the idea is 
# --> setup neural network's layers (how many, how big, etc) in keras
# --> initiate a model object with those layers
# --> compile the model specifying loss and optimizer functions
# --> run the model on training data 
# --> evaluate output

# main function to convert
def cels_to_fahr(cels):
    return 1.8*cels + 32


# get random celsius values from 0 to 100
celsius_q = 100*np.random.rand(500, 1)

# convert efficiently to fahrenheit. we now have training data with labels
fahrenheit_a = np.apply_along_axis(cels_to_fahr, 0, celsius_q)

# setup neural network layer. units is number of neurons and 
# input shape is the shape of the data, in this case 1x1
layer_0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# initialize model & setup our layer
model = tf.keras.Sequential([layer_0])

# compile the model, try different learning rates to have a feel for 
# performance 
learn_rate = 0.1
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(learn_rate))

# train the model
epochs = 500 # how many steps the model takes
trained_model = model.fit(celsius_q, fahrenheit_a, epochs=epochs, verbose=False)

# have a look at how well the model performed using seaborn
# seaborn expects a dataframe so we build it
epoch_label = [x for x in range(epochs)]
loss = trained_model.history['loss']

loss_df = pd.DataFrame(list(zip(loss, epoch_label)),
                       columns=['Loss', 'Epoch'])

sns.set(style="darkgrid")
sns.lineplot(x="Epoch", y="Loss", data=loss_df)


