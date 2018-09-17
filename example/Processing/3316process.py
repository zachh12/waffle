import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import h5py

import pygama.processing as pp
import pygama.decoders as dl
import pygama.calculators as pc
import pygama.transforms as pt

#path to the DaqTest_Run1 file
lanl_data_dir = os.path.join(os.getenv("DATADIR", "."), "mjd")
raw_data_dir = os.path.join(lanl_data_dir,"raw")

def main():
    runNumber = 6
    n_max = np.inf #max number of events to decode

    process_t0(runNumber, n_max=n_max)
    #process_t1(runNumber)
    #plot_waveforms(runNumber, num_waveforms=1)
    #plt.show()
    #plot_spectrum(runNumber)
    plt.show()

def plot_waveforms(runNumber, num_waveforms=5):
    path = os.path.join(os.getenv("DATADIR", "."), "mjd/t1/")
    file_name = "t1_run{}.h5".format(runNumber)
    file_name = os.path.join(path, file_name)
    dcdr = dl.SIS3316Decoder(file_name)
    df_events = pd.read_hdf(file_name, key=dcdr.decoder_name)
    plt.figure()
    plt.xlabel("Time [ns]")
    plt.ylabel("ADC [arb]")
    #plt.xlim(0, 2000)
    #plt.ylim(-20000, -10000)
    for i, (index, row) in enumerate(df_events.iterrows()):
        wf = dcdr.parse_event_data(row)

        plt.plot(wf.data)
        if i >=num_waveforms : break

def process_t0(runNumber, n_max=5000):
    lanl_data_dir = os.path.join(os.getenv("DATADIR", "."), "mjd")
    raw_data_dir = os.path.join(lanl_data_dir,"raw")

    runList = [runNumber]
    pp.process_tier_0(raw_data_dir, runList, output_dir=os.path.join(os.getenv("DATADIR", "."), "mjd/t1"), n_max=n_max)

def process_t1(runNumber):
    runList = [runNumber]
    pp.process_tier_1(os.path.join(os.getenv("DATADIR", "."), "mjd/t1"), runList, make_processor_list())

def plot_spectrum(runNumber):
    file_name = "t2_run{}.h5".format(runNumber)

    df = pd.read_hdf(file_name)
    plt.hist(df.energy, bins=200)
    plt.xlabel("Energy [keV]")
    plt.ylabel("Counts")

def make_processor_list():

    #Make a list of processors to do to the data for the "tier one" (ie, gatified)
    procs = pp.TierOneProcessorList()

    #pass energy thru to t1
    procs.AddFromTier0("channel")
    procs.AddFromTier0("energy", "onboard_energy")

    #baseline remove
    procs.AddCalculator(pc.fit_baseline, {"end_index":700}, output_name=["bl_slope", "bl_int"])
    procs.AddTransform(pt.remove_baseline, {"bl_0":"bl_int", "bl_1":"bl_slope"}, output_waveform="blrm_wf")

    #energy estimator: pz correct, calc trap
    procs.AddTransform(pt.pz_correct, {"rc":72}, input_waveform="blrm_wf", output_waveform="pz_wf")
    procs.AddTransform(pt.trap_filter, {"rampTime":200, "flatTime":400}, input_waveform="pz_wf", output_waveform="trap_wf")

    procs.AddCalculator(pc.trap_max, {}, input_waveform="trap_wf", output_name="trap_max")
    procs.AddCalculator(pc.trap_max, {"method":"fixed_time","pickoff_sample":400}, input_waveform="trap_wf", output_name="trap_ft")
    
    return procs

if __name__=="__main__":
    main()