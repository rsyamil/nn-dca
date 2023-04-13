import numpy as np
import util

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Reshape, LeakyReLU, Lambda
from tensorflow.keras.layers import Dense, Dropout, Flatten, Concatenate
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import regularizers, activations, initializers, constraints
from tensorflow.keras.constraints import Constraint

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import History 

from IPython.display import clear_output

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from tensorflow.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import get_custom_objects

import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import matplotlib.cm as cm
from matplotlib.colors import Normalize

#fix number of batch and timesteps
def hyperbolic_function(x):
    
	#print("Value x: %s" % K.get_value(x))
	#n_batch = K.get_value(tf.shape(x)[0])
	n_batch = 10
	n_timesteps = 60

	#comes as [[qi, b, di]], i.e. shape (Nb,3)
	qi = K.expand_dims(tf.gather(x, 0, axis=1), 0)
	b = K.expand_dims(tf.gather(x, 1, axis=1), 0)
	di = K.expand_dims(tf.gather(x, 2, axis=1), 0)
	t = K.expand_dims(K.constant(np.arange(0, n_timesteps)), 0)

	#repeat shapes for element-wise multiplication
	qi = K.transpose(K.repeat_elements(qi, n_timesteps, axis=0))
	b = K.transpose(K.repeat_elements(b, n_timesteps, axis=0))
	di = K.transpose(K.repeat_elements(di, n_timesteps, axis=0))
	t = K.repeat_elements(t, n_batch, axis=0)

	#print("Value qi: %s" % K.get_value(qi))
	#print("Value b: %s" % K.get_value(b))
	#print("Value di: %s" % K.get_value(di))
	#print("Value t: %s" % K.get_value(t))

	#print("Shape qi: %s" % qi)
	#print("Shape b: %s" % b)
	#print("Shape di: %s" % di)
	#print("Shape t: %s" % t)

	#hyperbolic equation
	output = qi/((1.0+b*di*t)**(1.0/b))
	output = K.expand_dims(output, -1)
	print("Shape output: %s" % output)
	return output

get_custom_objects().update({'hyperbolic_function': Lambda(hyperbolic_function)})

#test_input = K.constant([[0.833, 0.869, 0.725],
#                         [0.876, 0.965, 0.525],
#                         [0.559, 0.605, 0.760],
#                         [0.773, 0.644, 0.795],
#                         [0.975, 0.591, 0.534],
#                         [0.560, 0.961, 0.930],
#                         [0.825, 0.749, 0.719],
#                         [0.993, 0.702, 0.913],
#                         [0.626, 0.774, 0.708],
#                         [0.949, 0.952, 0.631]])

#test = hyperbolic_function(test_input)
#print(test)
#print(K.get_value(test))

def RMSE(x, y):
	return np.sqrt(np.mean(np.square(x.flatten() - y.flatten())))
    
class FCNN:

	def __init__(self, X, Y, n_P=3, name=[]):
    
		self.name = name

		self.X = X
		self.Y = Y
		self.n_P = n_P		#no of input parameters to equation

		self.F1F2 = []
		
		self.X2P_oil = []
		self.X2P_water = []
		self.X2P_gas = []
		self.P2Y = []
		
	def get_F1F2_model(self):
	
		input_x = Input(shape=(self.X.shape[1], ))
		_ = Dense(8)(input_x)
		_ = LeakyReLU(alpha=0.2)(_)
		_ = Dense(12)(_)
		_ = LeakyReLU(alpha=0.2)(_)
		_ = Dense(16)(_)
		_ = LeakyReLU(alpha=0.2)(_)
		
		P_oil = Dense(self.n_P, activation="sigmoid")(_)
		output_oil = Lambda(hyperbolic_function)(P_oil)
		
		P_water = Dense(self.n_P, activation="sigmoid")(_)
		output_water = Lambda(hyperbolic_function)(P_water)

		P_gas = Dense(self.n_P, activation="sigmoid")(_)
		output_gas = Lambda(hyperbolic_function)(P_gas)
		
		#concatenate all the predicted phases
		output_f1f2_y = Concatenate(axis=2)([output_oil, output_water, output_gas])
		
		return input_x, output_f1f2_y
		
	def train(self, n_batch=10, nb_epoch=100, load=False):

		#train F1F2 model
		input_x, output_f1f2_y = self.get_F1F2_model()
		self.F1F2 = Model(input_x, output_f1f2_y)
		
		self.F1F2.compile(loss='mse', optimizer=Adam(lr=1e-3))
		self.F1F2.summary()
		plot_model(self.F1F2, to_file='readme/'+self.name+'_f1f2_arch.png', show_shapes=True)
		
		if not load:
			history = History()
			losses = np.zeros([nb_epoch, 2])
        
			#set val split so that the batches are always 10 each
			for i in tqdm(range(nb_epoch)):
				self.F1F2.fit(self.X, self.Y, validation_split=0.0, 
						epochs=1, batch_size=n_batch, verbose=False,
						shuffle=False, callbacks=[history])
            
				losses[i, :]  = np.asarray(list(history.history.values()))[:, i]
			
				print ("%d [Loss: %f] [Val loss: %f]" % (i, losses[i, 0], losses[i, 1]))
            
				figs = util.plotLosses(losses, name="readme/"+self.name+"_f1f2_losses")
			self.F1F2.save('readme/'+self.name+'_f1f2_model.h5')
		else:
			print("Trained F1F2 model loaded")
			self.F1F2 = load_model('readme/'+self.name+'_f1f2_model.h5')
			
		#map x to P_oil
		x_oil = Input(shape=(self.X.shape[1], ))
		_ = self.F1F2.layers[1](x_oil)
		for i in [2, 3, 4, 5, 6]:
			_ = self.F1F2.layers[i](_)
		p_oil = self.F1F2.layers[7](_)
		self.X2P_oil = Model(x_oil, p_oil)
		
		#map x to P_water
		x_water = Input(shape=(self.X.shape[1], ))
		_ = self.F1F2.layers[1](x_water)
		for i in [2, 3, 4, 5, 6]:
			_ = self.F1F2.layers[i](_)
		p_water = self.F1F2.layers[8](_)
		self.X2P_water = Model(x_water, p_water)
		
		#map x to P_gas
		x_gas = Input(shape=(self.X.shape[1], ))
		_ = self.F1F2.layers[1](x_gas)
		for i in [2, 3, 4, 5, 6]:
			_ = self.F1F2.layers[i](_)
		p_gas = self.F1F2.layers[9](_)
		self.X2P_gas = Model(x_gas, p_gas)	
		
		#map any p to y (the empirical equation as a layer)
		f1f2_p_ = Input(shape=(self.n_P, ))
		f1f2_y = self.F1F2.layers[10](f1f2_p_)
		self.P2Y = Model(f1f2_p_, f1f2_y)
		




