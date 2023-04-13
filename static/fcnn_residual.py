import numpy as np
import util

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Reshape, LeakyReLU
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import regularizers, activations, initializers, constraints
from tensorflow.keras.constraints import Constraint

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import History 

from IPython.display import clear_output

from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

import string
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm

import matplotlib.cm as cm
from matplotlib.colors import Normalize

def RMSE(x, y):
	return np.sqrt(np.mean(np.square(x.flatten() - y.flatten())))
    
class FCNN:

	def __init__(self, X, Y, name=[]):
    
		self.name = name

		self.X = X
		self.Y = Y

		self.F3 = []
		
	def get_F3_model(self):
	
		input_x = Input(shape=(self.X.shape[1], ))
		_ = Dense(8)(input_x)
		_ = LeakyReLU(alpha=0.2)(_)
		
		_ = Dense(12)(_)
		_ = LeakyReLU(alpha=0.2)(_)
		_ = Dense(16)(_)
		_ = LeakyReLU(alpha=0.2)(_)
		
		_ = Dense(240)(_)
		_ = Reshape((15, 16))(_)

		_ = Conv1D(8*2, 6, padding="same")(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = UpSampling1D(2)(_)

		_ = Conv1D(16*2, 3, padding="same")(_)
		_ = LeakyReLU(alpha=0.3)(_)
		_ = UpSampling1D(2)(_)

		output_f3_y = Conv1D(3, 3, padding='same')(_)
		
		return input_x, output_f3_y
		
	def train(self, n_batch=10, nb_epoch=100, load=False):

		#train F3 model
		input_x, output_f3_y = self.get_F3_model()
		self.F3 = Model(input_x, output_f3_y)
		
		self.F3.compile(loss='mse', optimizer=Adam(lr=1e-3))
		self.F3.summary()
		plot_model(self.F3, to_file='readme/'+self.name+'_f3_arch.png', show_shapes=True)
		
		if not load:
			history = History()
			losses = np.zeros([nb_epoch, 2])
        
			for i in tqdm(range(nb_epoch)):
				self.F3.fit(self.X, self.Y, validation_split=0.2, 
						epochs=1, batch_size=n_batch, verbose=False,
						shuffle=False, callbacks=[history])
            
				losses[i, :]  = np.asarray(list(history.history.values()))[:, i]
			
				print ("%d [Loss: %f] [Val loss: %f]" % (i, losses[i, 0], losses[i, 1]))
            
				figs = util.plotLosses(losses, name="readme/"+self.name+"_f3_losses")
			self.F3.save('readme/'+self.name+'_f3_model.h5')
		else:
			print("Trained F3 model loaded")
			self.F3 = load_model('readme/'+self.name+'_f3_model.h5')
			


