from models import neuralNetwork as nn
import tensorflow as tf 
import numpy as np
optimizer=tf.keras.optimizers.Adam(1e-3)
loss='mse'
metrics=['mse']
neuralNetwork=nn.trainNn(np.random.randint(low=0,high=100,size=(20,2)),np.random.randint(low=0,high=100,size=20),optimizer=optimizer,loss=loss,metrics=metrics)