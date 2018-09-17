import numpy as np
import matplotlib.pyplot as plt
from waffle.management import WaveformFitManager
import waffle.management
from waffle.models import VelocityModel, LowPassFilterModel, HiPassFilterModel, ImpurityModelEnds
from siggen import PPC
import pickle
import os
from waffle.management import FitConfiguration

sample = np.load("fit_params.npy")
print(list(sample))
print(sample['model_conf'][3])
quit()





det_params = [ 9.76373631e-01,8.35875049e-03,-5.09732644e+00,-6.00749043e+00,
                   4.74275220e+06,3.86911389e+06,6.22014783e+06,5.22077471e+06,
                    -3.63516477e+00,-4.48184667e-01]
print(det_params[:2],
det_params[2:4],
det_params[4:8],
det_params[8:])

    



