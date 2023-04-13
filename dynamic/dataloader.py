import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as m

import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper", rc={"font.size":14,"axes.titlesize":14,"axes.labelsize":14})  

#plot multiphase profile
def plot_production(y_train, y_test, name=""):
    timesteps = np.linspace(0, y_train.shape[1]-1, y_train.shape[1])
    f_names = ["Oil", "Water", "Gas"]
    colors = ['g','b','r'] 
    
    fig = plt.figure(figsize=(12, 3.5))
    
    for i in range(len(f_names)):
        ax = plt.subplot(1, 3, i+1)    
        
        for k in range(y_train.shape[0]):
            ax.plot(timesteps, y_train[k, :, i], alpha=0.1, color=colors[i])
        for k in range(y_test.shape[0]):
            ax.plot(timesteps, y_test[k, :, i], alpha=0.6, color=colors[i])
                
        ax.set_xlim([0, y_train.shape[1]-1])
        ax.set_ylabel("Rate (bpd)")
        ax.set_xlabel("Timesteps")
        ax.set_title(f_names[i])
        ax.grid(False)
    fig.tight_layout() 
    fig.savefig('readme/'+name+'-prod.png')

#function to plot the density plot
def histplot(data_train, data_test, name=""):
	col_names = ['Perm. (mD)', 'Poro. (ratio)', 'SW (ratio)', 'Thick. (m)', 'Press. (psi)', 'Density (kgm3)']
	plt.figure(figsize=(9, 6))
	for idx, feature in enumerate(col_names):
		plt.subplot(2, 3, idx+1)
		dtr = data_train[:, idx]
		dts = data_test[:, idx]
		dtr = dtr[~np.isnan(dtr)]	#drop existing nans
		dts = dts[~np.isnan(dts)]
		new_bins = np.linspace(np.min(dtr), np.max(dtr), 30)
		sns.distplot(dtr, hist=True, bins=new_bins, norm_hist=False, kde=False, label="Train")
		sns.distplot(dts, hist=True, bins=new_bins, norm_hist=False, kde=False, label="Test")
		plt.tick_params(axis='both', which='both', bottom='on', top='off', labelbottom='on', right='off', left='on', labelleft='off')
		plt.tight_layout(), plt.legend(), plt.title(feature)
	plt.savefig('readme/'+name+'-prod.png')
	
def simulate_field_dataset(Y):
	"""
	Arguments:
		Y: Float. Matrix of multiphase production data
	Output: 
	Returns Y_field, or the matrix of partially observed multiphase production data. Float.
	"""
	n_row, n_col, n_channel = Y.shape
	
	np.random.seed(18)
	C_timesteps = np.random.random_integers(low=1, high=int(n_col/1.5), size=n_row)
	
	C = np.ones(Y.shape)
	for c_timesteps, c in zip(C_timesteps, C):
		c[-c_timesteps:, :] = 0
		
	Y_masked = Y.copy()
	Y_masked[C == 0] = np.nan
	
	return Y_masked, C

class DataLoader:

	def __init__(self, dataset_type="GEOMECH", verbose=False):

		self.verbose = verbose
		self.dataset_type = dataset_type

		self.x = []         	#(800, 7)
		self.y = []         	#(800, 60, 3)
		self.y_cumm = []		#(800, 1)
		
		self.x_raw = []         #(800, 7)
		self.y_raw = []         #(800, 60, 3)
		self.y_cumm_raw = []	#(800, 1)

		self.x_min = 0
		self.x_max = 0
		self.y_min = np.array ([])   #(3,) for each channel
		self.y_max = np.array ([])   #(3,)
		
		self.x_means = []
        
	def normalize_x(self):
		self.x_min = np.min(self.x, axis=0)
		self.x_max = np.max(self.x, axis=0)
		self.x = (self.x - self.x_min)/(self.x_max - self.x_min)
    
	def normalize_y(self):
		'''normalize by channel'''
		n_features = self.y.shape[-1]
		for f in range(n_features):
			self.y_min = np.append(self.y_min, np.min(self.y[:,:,f]))
			self.y_max = np.append(self.y_max, np.max(self.y[:,:,f]))
			self.y[:,:,f] = (self.y[:,:,f] - self.y_min[f])/(self.y_max[f] - self.y_min[f])

	def generate_data(self, seed=888, simulate_field=False):
    
		if self.dataset_type == "GEOMECH":
			df = pd.read_csv("Data_Simulated_Bakken/DATA_BAKKEN_GEOMECH.csv")
		else:
			df = pd.read_csv("Data_Simulated_Bakken/DATA_BAKKEN_NFR.csv")
		df = df.to_numpy()

		#original data
		self.x = df[:, 0:7]
		oil = df[:, 7:7+60]
		water = df[:, 7+60:7+60+60]
		gas = df[:, 7+60+60:7+60+60+60]
		self.y = np.stack((oil, water, gas), axis=2)
		
		#calculate cumulative oil
		self.y_cumm_raw = np.sum(oil, axis=1)
		self.y_cumm = (self.y_cumm_raw - np.min(self.y_cumm_raw))/(np.max(self.y_cumm_raw) - np.min(self.y_cumm_raw))

		#shuffle x and y together, since theyre from the same provenance! 
		np.random.seed(77)
		shuffle_idx = np.random.permutation(self.x.shape[0])

		#shuffle data
		self.x = self.x[shuffle_idx]
		self.y = self.y[shuffle_idx]

		#make copies (for spatial plotting) and normalize
		self.x_raw = np.copy(self.x)
		self.normalize_x()

		self.y_raw = np.copy(self.y)
		self.normalize_y()
		
		#calculate the mean
		self.x_means = np.mean(self.x, axis=0)

		#randomly remove trailing values to simulate field data
		if simulate_field:
			y_masked, c = simulate_field_dataset(self.y)
			return self.x, self.y, y_masked, c

		return self.x, self.y

    
        

    
    
    