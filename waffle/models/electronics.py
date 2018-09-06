import os, sys
import numpy as np
from scipy import signal

from ._parameterbase import JointModelBase, Parameter
from siggen.electronics import DigitalFilter, GretinaOvershootFilter

class DigitalFilterModel(JointModelBase):
    def __init__(self, detector, order, include_zeros, pmag_lims=None, pphi_lims=None, zmag_lims=None, zphi_lims=None, do_transform=False):
        assert (order==1 or order==2)

        if pmag_lims is None: pmag_lims = [0,1]
        if zmag_lims is None: zmag_lims = [0,1]
        if pphi_lims is None: pphi_lims = [0,np.pi]
        if zphi_lims is None: zphi_lims = [0,np.pi]

        self.order = order
        self.include_zeros = include_zeros
        self.do_transform = do_transform

        self.digital_filter = DigitalFilter(order)
        detector.AddDigitalFilter(self.digital_filter)

        self.freq_samp = 1E9
        self.nyq_freq = 0.5*self.freq_samp

        if self.order == 1:
            self.params = [
                Parameter("pole_mag", "uniform", lim_lo=pmag_lims[0], lim_hi=pmag_lims[-1]),
            ]

            if include_zeros:
                self.params.append(
                    Parameter("zero_mag", "uniform", lim_lo=zmag_lims[0], lim_hi=zmag_lims[-1]),
                )

        elif self.order == 2:
            self.params = [
                #I know from experience that the lowpass poles are near (0,1)
                #(makes sense cause the amplitude response should fall off near nyquist freq)
                #just go ahead and shove the priors up near there
                Parameter("pole_mag", "uniform", lim_lo=pmag_lims[0], lim_hi=pmag_lims[-1]),
                Parameter("pole_phi", "uniform", lim_lo=pphi_lims[0], lim_hi=pphi_lims[-1]),
            ]

            if include_zeros:
                self.params.append(
                    Parameter("zero_mag", "uniform", lim_lo=zmag_lims[0], lim_hi=zmag_lims[-1]),
                )
                self.params.append(
                    Parameter("zero_phi", "uniform", lim_lo=zphi_lims[0], lim_hi=zphi_lims[-1]),
                )

    def get_pz(self, params):
        self.apply_to_detector(params, None)

        return_dict = {
            "poles":[],
            "zeros":[]
        }

        try:
            return_dict["poles"].append(self.digital_filter.poles)
        except AttributeError:
            pass

        try:
            return_dict["zeros"].append(self.digital_filter.zeros)
        except AttributeError:
            pass


        return return_dict

    def apply_to_signal(self, signal):
        return self.digital_filter.apply_to_signal(signal)

    def apply_to_detector(self, params, detector):
        if self.order == 1:
            pole_mag, pole_phi = params[0], 0
            zero_mag = params[1] if self.include_zeros else None
            zero_phi = 0
        else:
            (pole_mag, pole_phi) = params[:2]
            zero_mag= params[2] if self.include_zeros else None
            zero_phi= params[3] if self.include_zeros else None

        if self.do_transform:
            pole_mag, pole_phi, zero_mag, zero_phi = self.transform_params(pole_mag, pole_phi, zero_mag, zero_phi)

        self.digital_filter.set_poles(pole_mag, pole_phi)

        if self.include_zeros:
            self.digital_filter.set_zeros(zero_mag, zero_phi)

    def get_freqz(self, params, w):
        self.apply_to_detector(params, None)
        # w=np.logspace(-15, -5, 500, base=np.pi)

        num, den = self.digital_filter.num, self.digital_filter.den
        if np.sum(num) != 0: num_calc = num/(np.sum(num)/np.sum(den))
        else: num_calc=num
        w, h = signal.freqz(num_calc, den, worN=w )
        w/= (np.pi /self.nyq_freq)
        return w, h

class LowPassFilterModel(DigitalFilterModel):
    def __init__(self,detector, order=2, include_zeros=False, pmag_lims=None, pphi_lims=None, zmag_lims=None, zphi_lims=None):
        if pmag_lims is None: pmag_lims = [0.9,1]
        if pphi_lims is None: pphi_lims = [0,0.1]
        # if zmag_lims is None: zmag_lims = [0.9,1]
        # if zphi_lims is None: zphi_lims = [0,0.5]

        super().__init__(detector, order, include_zeros, pmag_lims, pphi_lims, zmag_lims, zphi_lims)

        if not self.include_zeros:
            if order == 1:self.digital_filter.num = [1,1]
            else: self.digital_filter.num = [1,2,1]


class HiPassFilterModel(DigitalFilterModel):
    def __init__(self, detector, order=1, include_zeros=False, pmag_lims=None, pphi_lims=None):
        assert include_zeros == False
        order = 1
        assert order == 1

        if pmag_lims is None: pmag_lims = [60,80]
        # if pmag_lims is None: pmag_lims = [-6,-1]
        # if pphi_lims is None: pphi_lims = [-20,-2]

        super().__init__(detector, order, include_zeros, pmag_lims, pphi_lims, None, None, do_transform=True)

        if order == 1:
            self.digital_filter.num = [1,-1]
        else:
            self.digital_filter.num = [1,-2,1]

    def transform_params(self, pmag, pphi, zmag, zphi):
        pmag = np.exp(-1./self.freq_samp/(pmag*1E-6))
        # pmag = 1. - 10.**pmag
        if pphi != 0: pphi = np.pi**pphi

        return (pmag, pphi, None, None)

class OvershootFilterModel(DigitalFilterModel):
    '''
    First order filter with one pole one zero
    Two parameters:
    --Zero location, in us
    --Pole location, relative to zero, in log
    '''
    def __init__(self, detector, pmag_lims=None, zmag_lims=None):
        order = 1
        include_zeros = True

        if zmag_lims is None: zmag_lims = [0.1,10] #us decay constant (should be around 2?)
        if pmag_lims is None: pmag_lims = [-8,-3]

        super().__init__(detector, order, include_zeros, pmag_lims, None, zmag_lims, None, do_transform=True)

    def transform_params(self, pmag, pphi, zmag, zphi):
        zmag = np.exp(-1./self.freq_samp/(zmag*1E-6))
        pmag = zmag - 10.**pmag

        return (pmag, 0., zmag, 0.)

class OscillationFilterModel(DigitalFilterModel):
    '''
    Second order filter, Two parameters:
    --Pole magnitude is (inverse) strength (amplitude) of oscillation
    --complex angle is oscillation frequency in MHz
    --zero is relative to the pole
    Transfer function is:
                1
        -----------------
            (1-p)(1-p*)
    where p = mag*exp(j*phi)
    '''
    def __init__(self, detector, include_zeros = True, pmag_lims=None, pphi_lims=None, zmag_lims=None, zphi_lims=None):
        order = 2

        if pmag_lims is None: pmag_lims = [-4,0]
        if pphi_lims is None: pphi_lims = [3, 7]
        if zmag_lims is None: zmag_lims = [-4,0]
        if zphi_lims is None: zphi_lims = [3, 7]

        super().__init__(detector, order, include_zeros, pmag_lims, pphi_lims, zmag_lims, zphi_lims, do_transform=True)

    def transform_params(self, pmag, pphi, zmag, zphi):
        pmag = 1. - 10.**pmag
        pphi = pphi*1E6 * (np.pi /self.nyq_freq)

        zmag = 1. - 10.**zmag

        zphi = pphi

        return (pmag, pphi, zmag, zphi)

class AntialiasingFilterModel(JointModelBase):
    '''
    Fourth order lowpass designed to look like simulated model of Gretina input
    '''

    def __init__(self,detector):
        self.freq_samp = 1E9
        self.nyq_freq = 0.5*self.freq_samp

        self.params = [
            Parameter("aa_mag", "uniform", lim_lo=0.5, lim_hi=1),
            Parameter("aa_phi", "uniform", lim_lo=0.0, lim_hi=0.25),
        ]

        self.digital_filter1 = DigitalFilter(2)
        self.digital_filter2 = DigitalFilter(2)
        detector.AddDigitalFilter(self.digital_filter1)
        detector.AddDigitalFilter(self.digital_filter2)

        self.digital_filter1.num = [1,1]
        self.digital_filter2.num = [1]

    def apply_to_detector(self, params, detector):
        mag, phi = params[:]
        self.digital_filter1.set_poles(mag, phi)
        self.digital_filter2.set_poles(mag, phi)

    def get_freqz(self, params, w):
        self.apply_to_detector(params, None)
        # w=np.logspace(-15, -5, 500, base=np.pi)

        num1, den1 = self.digital_filter1.num, self.digital_filter1.den
        num1_c = num1/ (np.sum(num1)/np.sum(den1))
        w, h1 = signal.freqz(num1_c, den1, worN=w )

        num2, den2 = self.digital_filter2.num, self.digital_filter2.den
        num2_c = num2/ (np.sum(num2)/np.sum(den2))
        w, h2 = signal.freqz(num2_c, den1, worN=w )

        w/= (np.pi /self.nyq_freq)

        return w, h1*h2

    def apply_to_signal(self, signal):
        sig = self.digital_filter1.apply_to_signal(signal)
        return self.digital_filter2.apply_to_signal(sig)

class FirstStageFilterModel(JointModelBase):
    '''
    Fourth order lowpass designed to look MJD first stage
    '''

    def __init__(self,detector):
        self.freq_samp = 1E9
        self.nyq_freq = 0.5*self.freq_samp
        self.params = [
            Parameter("fs_fo_pole", "uniform", lim_lo=-2.5, lim_hi=-0.5),
            Parameter("fs_fo_zero", "uniform", lim_lo=0.7, lim_hi=1),
            # Parameter("fs_so_mag", "uniform", lim_lo=pmag_lims[0], lim_hi=pmag_lims[-1]),
            Parameter("fs_so_phi", "uniform", lim_lo=-8, lim_hi=0),
        ]

        self.digital_filter1 = DigitalFilter(1)
        self.digital_filter2 = DigitalFilter(2)
        detector.AddDigitalFilter(self.digital_filter1)
        detector.AddDigitalFilter(self.digital_filter2)

        self.digital_filter2.num = [1]


    def apply_to_detector(self, params, detector):
        fo_pole, fo_zero, so_phi  = params[:]

        fo_pole = 1- 10**fo_pole
        so_phi = 10**so_phi

        self.digital_filter1.set_poles(fo_pole)
        self.digital_filter1.set_zeros(fo_zero)

        self.digital_filter2.set_poles(fo_zero, so_phi)

    def get_freqz(self, params, w):
        self.apply_to_detector(params, None)
        # w=np.logspace(-15, -5, 500, base=np.pi)

        num1, den1 = self.digital_filter1.num, self.digital_filter1.den
        num1_c = num1/ (np.sum(num1)/np.sum(den1))
        w, h1 = signal.freqz(num1_c, den1, worN=w )

        num2, den2 = self.digital_filter2.num, self.digital_filter2.den
        num2_c = num2/ (np.sum(num2)/np.sum(den2))
        w, h2 = signal.freqz(num2_c, den1, worN=w )

        w/= (np.pi /self.nyq_freq)

        return w, h1*h2
