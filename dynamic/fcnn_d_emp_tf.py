import numpy as np
import util

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Flatten, LeakyReLU
from tensorflow.keras.layers import Input, Reshape, Dense, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D

from tensorflow.keras.layers import Conv1D, UpSampling1D
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D, GlobalAveragePooling1D

from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention

from tensorflow.keras.layers import LSTM, Dropout, Concatenate, TimeDistributed

from tensorflow.keras import backend as K

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras import regularizers, activations, initializers, constraints
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.callbacks import History, EarlyStopping

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

from tensorflow.python.keras.utils.generic_utils import get_custom_objects

import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import matplotlib.cm as cm
from matplotlib.colors import Normalize

N_TIMESTEPS = 60

#fix number of batch and timesteps
def hyperbolic_function(x):
    
	n_batch = 10
	n_timesteps = N_TIMESTEPS

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

	#hyperbolic equation
	output = qi/((1.0+b*di*t)**(1.0/b))
	output = K.expand_dims(output, -1)
	return output

get_custom_objects().update({'hyperbolic_function': Lambda(hyperbolic_function)})

def RMSE(x, y):
	return np.sqrt(np.mean(np.square(x.flatten() - y.flatten())))
	
#ignores the nan when calculating RMSE
def nanRMSE(data1, data2):
    data1[data1 == -1] = np.nan
    data2[data1 == -1] = np.nan
    return np.sqrt(np.nanmean((data1.flatten()-data2.flatten())**2))
    
class FCNN:

	def __init__(self, X, Y, Y_and_mask, n_P=3, name=[]):
    
		self.name = name

		self.X = X
		self.Y = Y
		self.Y_and_mask = Y_and_mask
		self.n_P = n_P		#no of input parameters to equation

		self.F1F2 = []
		
		self.X2P_oil = []
		self.X2P_water = []
		self.X2P_gas = []
		self.P2Y = []
		
		self.X2A_1 = []
		self.X2A_2 = []
		
		self.X2AV_1 = []
		self.X2AV_2 = []
		
		self.n_batch = 0
		self.nb_epoch = 0
		
		self.num_transformer_blocks = 2
        
		self.head_size = 1
		self.num_heads = 16
		self.ff_dim = 16
		self.dropout = 0.4
		self.mlp_dropout = 0.25
		
	def get_tf_prop_encoder(self, input, head_size, num_heads, ff_dim, dropout=0):
	
		_ = LayerNormalization(epsilon=1e-6)(input)
		_, weights = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(_, _, return_attention_scores=True)
		_ = Dropout(dropout)(_)
		res = _ + input

		_ = LayerNormalization(epsilon=1e-6)(res)
		_ = Dense(input.shape[-1])(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = Dense(input.shape[-1])(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = Dense(input.shape[-1])(_)
		return _ + res, weights
		
	def get_F1F2_model(self):
	
		input_x = Input(batch_shape=(self.n_batch, self.X.shape[1], self.X.shape[2]))
		_ = input_x
		
		'''
		for n in range(self.num_transformer_blocks):
			_ = self.get_tf_prop_encoder(_, self.head_size, self.num_heads, self.ff_dim, self.dropout)
		x_encoding = Flatten()(_)
		'''
		
		activ_1, scores_1 = self.get_tf_prop_encoder(_, self.head_size, self.num_heads, self.ff_dim, self.dropout)
		activ_2, scores_2 = self.get_tf_prop_encoder(activ_1, self.head_size, self.num_heads, self.ff_dim, self.dropout)
		x_encoding = Flatten()(activ_2)
		
		_ = Dense(9)(x_encoding)
		_ = LeakyReLU(alpha=0.2)(_)

		P_oil = Dense(self.n_P, activation="sigmoid")(_)
		output_oil = Lambda(hyperbolic_function)(P_oil)
		
		P_water = Dense(self.n_P, activation="sigmoid")(_)
		output_water = Lambda(hyperbolic_function)(P_water)

		P_gas = Dense(self.n_P, activation="sigmoid")(_)
		output_gas = Lambda(hyperbolic_function)(P_gas)
		
		#concatenate all the predicted phases
		output_f1f2_y = Concatenate(axis=2)([output_oil, output_water, output_gas])
		
		return input_x, output_f1f2_y, P_oil, P_water, P_gas, scores_1, scores_2, activ_1, activ_2
		
	def train(self, n_batch=10, nb_epoch=100, load=False):

		self.n_batch = n_batch
		self.nb_epoch = nb_epoch
		
		#train F1F2 model
		input_x, output_f1f2_y, P_oil, P_water, P_gas, scores_1, scores_2, activ_1, activ_2 = self.get_F1F2_model()
		self.F1F2 = Model(input_x, output_f1f2_y)
		
		def custom_mse_loss(y_true_and_mask, y_pred):
			y_true = y_true_and_mask[:, :N_TIMESTEPS]
			mask = y_true_and_mask[:, N_TIMESTEPS:]
			y_pred_masked = tf.math.multiply(y_pred, mask)
			#the returned scalar does not ignore nan
			return tf.keras.losses.mean_squared_error(y_true = y_true, y_pred = y_pred_masked)
		#add custom loss 
		get_custom_objects().update({"custom_mse_loss": custom_mse_loss})
		
		self.F1F2.compile(loss=custom_mse_loss, optimizer=Adam(lr=1e-3))
		self.F1F2.summary()
		plot_model(self.F1F2, to_file='readme/'+self.name+'_f1f2_arch.png', show_shapes=True)
		
		if not load:
			history = History()
			losses = np.zeros([nb_epoch, 2])
        
			#set val split so that the batches are always 10 each
			for i in tqdm(range(nb_epoch)):
				self.F1F2.fit(self.X, self.Y_and_mask, validation_split=0.0, 
						epochs=1, batch_size=n_batch, verbose=False,
						shuffle=False, callbacks=[history])
            
				losses[i, :]  = np.asarray(list(history.history.values()))[:, i]
			
				print ("%d [Loss: %f] [Val loss: %f]" % (i, losses[i, 0], losses[i, 1]))
            
				figs = util.plotLosses(losses, name="readme/"+self.name+"_f1f2_losses")
			self.F1F2.save('readme/'+self.name+'_f1f2_model.h5')
		else:
			print("Trained F1F2 model loaded")
			self.F1F2 = load_model('readme/'+self.name+'_f1f2_model.h5', custom_objects={"custom_mse_loss": custom_mse_loss})
		
		input_x, output_f1f2_y, p_oil, p_water, p_gas, scores_1, scores_2, activ_1, activ_2 = self.get_F1F2_model()
		X2P_oil_idxs = np.append(np.arange(0, 26), [26])
		X2P_water_idxs = np.append(np.arange(0, 26), [27])
		X2P_gas_idxs = np.append(np.arange(0, 26), [28])
		
		X2A_1_idxs = np.arange(0, 3)
		X2A_2_idxs = np.arange(0, 14)
		
		#map x to P_oil
		self.X2P_oil = Model(input_x, p_oil)
		self.X2P_oil.summary()
		
		for j in range(len(X2P_oil_idxs)):
			self.X2P_oil.layers[j].set_weights(self.F1F2.layers[X2P_oil_idxs[j]].get_weights())
		
		#map x to P_water
		self.X2P_water = Model(input_x, p_water)
		self.X2P_water.summary()
		
		for j in range(len(X2P_water_idxs)):
			self.X2P_water.layers[j].set_weights(self.F1F2.layers[X2P_water_idxs[j]].get_weights())
		
		#map x to P_gas
		self.X2P_gas = Model(input_x, p_gas)
		self.X2P_gas.summary()
		
		for j in range(len(X2P_gas_idxs)):
			self.X2P_gas.layers[j].set_weights(self.F1F2.layers[X2P_gas_idxs[j]].get_weights())
			
		#map any p to y (the empirical equation as a layer)
		f1f2_p_ = Input(shape=(self.n_P, ))
		f1f2_y = self.F1F2.layers[29](f1f2_p_)
		self.P2Y = Model(f1f2_p_, f1f2_y)
		self.P2Y.summary()
		
		#get first multi-head attention score
		self.X2A_1 = Model(input_x, scores_1)
		self.X2A_1.summary()

		for j in range(len(X2A_1_idxs)):
			self.X2A_1.layers[j].set_weights(self.F1F2.layers[X2A_1_idxs[j]].get_weights())
		
		#get second multi-head attention score
		self.X2A_2 = Model(input_x, scores_2)
		self.X2A_2.summary()
		
		for j in range(len(X2A_2_idxs)):
			self.X2A_2.layers[j].set_weights(self.F1F2.layers[X2A_2_idxs[j]].get_weights())
			
		#get first multi-head attention activations
		self.X2AV_1 = Model(input_x, activ_1)
		self.X2AV_1.summary()
		
		for j in range(len(X2A_1_idxs)):
			self.X2AV_1.layers[j].set_weights(self.F1F2.layers[X2A_1_idxs[j]].get_weights())
		
		#get second multi-head attention activations
		self.X2AV_2 = Model(input_x, activ_2)
		self.X2AV_2.summary()
		
		for j in range(len(X2A_2_idxs)):
			self.X2AV_2.layers[j].set_weights(self.F1F2.layers[X2A_2_idxs[j]].get_weights())
		
		'''
		#map x to P_oil
		x_oil = Input(shape=(self.X.shape[1], self.X.shape[2]))
		_ = self.F1F2.layers[1](x_oil)
		for i in np.arange(2, 26):
			_ = self.F1F2.layers[i](_)
		p_oil = self.F1F2.layers[26](_)
		self.X2P_oil = Model(x_oil, p_oil)
		
		#map x to P_water
		x_water = Input(shape=(self.X.shape[1], self.X.shape[2]))
		_ = self.F1F2.layers[1](x_water)
		for i in np.arange(2, 26):
			_ = self.F1F2.layers[i](_)
		p_water = self.F1F2.layers[27](_)
		self.X2P_water = Model(x_water, p_water)
		
		#map x to P_gas
		x_gas = Input(shape=(self.X.shape[1], self.X.shape[2]))
		_ = self.F1F2.layers[1](x_gas)
		for i in np.arange(2, 26):
			_ = self.F1F2.layers[i](_)
		p_gas = self.F1F2.layers[28](_)
		self.X2P_gas = Model(x_gas, p_gas)	
		
		#map any p to y (the empirical equation as a layer)
		f1f2_p_ = Input(shape=(self.n_P, ))
		f1f2_y = self.F1F2.layers[29](f1f2_p_)
		self.P2Y = Model(f1f2_p_, f1f2_y)
		'''




