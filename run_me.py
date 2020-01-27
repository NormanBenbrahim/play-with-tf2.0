import tensorflow as tf
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

# formula for converting temperatures
def cels_to_fahr(cels):
    return 1.8*cels + 32

# read the file
temp_cels = pd.read_csv('data.csv')

# get temp_cels on planets in arrays
jupiter_cels = temp_cels['temp_jupiter'].values 
mars_cels = temp_cels['temp_mars'].values 
venus_cels = temp_cels['temp_venus'].values

# convert our data 
jupiter_fahr = np.apply_along_axis(cels_to_fahr, 0, jupiter_cels)
mars_fahr = np.apply_along_axis(cels_to_fahr, 0, mars_cels)
venus_fahr = np.apply_along_axis(cels_to_fahr, 0, venus_cels)

# setup simple neural network layer, 1 layer & 1 neuron
layer = tf.keras.layers.Dense(units=1, input_shape=[1])

# initialize the model & setup layer
model = tf.keras.Sequential([layer])
learn_rate = 0.1
epoch = 500
model.compile(loss='mean_squared_error',
             optimizer=tf.keras.optimizers.Adam(learn_rate))

# train the model
trained_model = model.fit(jupiter_cels, jupiter_fahr, epochs=epoch, verbose=False)

# plot the loss values using seaborn, seaborn expects a dataframe
loss = trained_model.history['loss']
epoch_label = [x for x in range(epoch)]
loss_df = pd.DataFrame(list(zip(loss, epoch_label)), columns=['Loss', 'Epoch'])

loss_df

sns.set(style="darkgrid")
sns.lineplot(x="Epoch", y="Loss", data=loss_df)
plt.show()