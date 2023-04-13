import keras
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

def plot_Y_res(Y, Y_hat, Y_hat_res, P, case=0, name=""):

	T = np.linspace(0, Y.shape[1]-1, Y.shape[1])
	colors = ['g','b','r'] 
	plt.figure(figsize=[10, 3])
	for i in range(Y.shape[2]):
		plt.subplot(1, 3, i+1)
		plt.scatter(T, Y[case, :, i], alpha=0.4, label='$y$', c='gray')
		plt.plot(T, Y_hat[case, :, i], alpha=0.9, linestyle="--", label='$\hat{y}$', c=colors[i]) 
		plt.plot(T, Y_hat_res[case, :, i], alpha=0.9, label='$\hat{y}-r$', c=colors[i]) 
		#plt.grid(False)
		plt.legend()
		plt.ylabel("Rate")
		plt.xlabel("Timesteps")
		plt.ylim(0, 1)
		#format title string
		title = "[ "
		for _ in P[case, :, i]:
			title = title + f"{_:.2f}" + " "
		plt.title(title+"]")
	plt.tight_layout()
	plt.savefig("readme/"+name+".png", dpi=400)

def plot_Y(Y, Y_hat, P, case=0, name=""):

	T = np.linspace(0, Y.shape[1]-1, Y.shape[1])
	colors = ['g','b','r'] 
	plt.figure(figsize=[10, 3])
	for i in range(Y.shape[2]):
		plt.subplot(1, 3, i+1)
		plt.scatter(T, Y[case, :, i], alpha=0.4, label='$y$', c='gray')
		plt.plot(T, Y_hat[case, :, i], alpha=0.9, label='$\hat{y}$', c=colors[i]) 
		#plt.grid(False)
		plt.legend()
		plt.ylabel("Rate")
		plt.xlabel("Timesteps")
		plt.ylim(0, 1)
		#format title string
		title = "[ "
		for _ in P[case, :, i]:
			title = title + f"{_:.2f}" + " "
		plt.title(title+"]")
	plt.tight_layout()
	plt.savefig("readme/"+name+".png", dpi=400)

def RMSE(data1, data2):
	return np.sqrt(np.mean((data1.flatten()-data2.flatten())**2))

#function to view training and validation losses
#function to view losses
def plotLosses(loss, name=[]):         
    
	fig = plt.figure(figsize=(6, 4))
	plt.plot(loss[:, 0], label='loss', c = 'green')
	plt.legend()

	plt.grid(False)
	plt.xlabel('Epoch')
	plt.ylabel('Loss')

	fig.tight_layout() 
	fig.savefig(name+'.png')
	plt.close(fig)
        
#function to view multiple losses
def plotAllLosses(loss1):         
	N, m1f = loss1.shape

	print(loss1.shape)

	fig = plt.figure(figsize=(6, 6))
	plt.subplot(2, 1, 1)
	plt.plot(loss1[:, 0], label='l1', linewidth=3)
	plt.plot(loss1[:, 1], label='l2')
	plt.legend()

	return fig
	
#scatter plots for training and testing, color by field label
def scatterPlot(data1, data2, xlabel, ylabel, color, title, ylim=0, name=""):
	fig, ax = plt.subplots(figsize=(4, 4))
	plt.scatter(data1, data2, c=color, alpha=0.2)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.xlim([0, ylim])
	plt.ylim([0, ylim])
	plt.title(title)
	ax.annotate(f"RMSE: {RMSE(data1, data2):.3f}", xy=(0.8, 0.8),  xycoords='data',
			xytext=(0.05, 0.9), textcoords='axes fraction',
			horizontalalignment='left', verticalalignment='top')
	print(RMSE(data1, data2))
	fig.savefig(name+'.png')