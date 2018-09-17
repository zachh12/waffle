import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from waffle.processing import *
import siggen
from siggen import PPC

def plot_train():
	det = PPC("config_files/p1_new.config")
	plt.plot(det.GetWaveform(20,0,3, 1200))
	df = np.load("../training_data/chan66_8wfs.npz")
	wf = df['wfs'][0]
	print(len(wf.window_waveform()))
	quit()
	plt.plot(wf.get_waveform() - np.mean(wf.get_waveform()[:30]))
	plt.xlim(0, 400)
	plt.show()

def main():
	#find_trap()
	plot_train()

if __name__ == "__main__":
    main()