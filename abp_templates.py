"""
Created on Tue Jul 17 21:31:09 2018

@author: phil

Waveform template viewer

Adapted heavily from biosppy.signals.ecg
Used to show summary waveforms for an entire segment in a single plot
Segments a hemodynamic waveform (eg ABP or CVP) based on ECG R-peak locations

This is meant to be used as a quick reference to determine if a segment is suitable for training/classification

We still use biosppy.signals.ecg for R peak detection

"""

import pandas as pd
import numpy as np
import waveform
from biosppy.signals import ecg

from bokeh.plotting import figure 
from bokeh.io import output_file, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource

def _extract_heartbeats(signal=None, rpeaks=None, before=200, after=400):
# taken from biosppy.signals.ecg
    
    """Extract heartbeat templates from an ECG signal, given a list of
    R-peak locations.
    Parameters
    ----------
    signal : array
        Input ECG signal.
    rpeaks : array
        R-peak location indices.
    before : int, optional
        Number of samples to include before the R peak.
    after : int, optional
        Number of samples to include after the R peak.
    Returns
    -------
    templates : array
        Extracted heartbeat templates.
    rpeaks : array
        Corresponding R-peak location indices of the extracted heartbeat
        templates.
    """

    R = np.sort(rpeaks)
    length = len(signal)
    templates = []
    newR = []

    for r in R:
        a = r - before
        if a < 0:
            continue
        b = r + after
        if b > length:
            break
        templates.append(signal[a:b])
        newR.append(r)

    templates = np.array(templates)
    newR = np.array(newR, dtype='int')

    return templates, newR

def extract_heartbeats(signal=None, rpeaks=None, sampling_rate=240.,
                       before=0.2, after=0.4):
    
# modified
    """Extract heartbeat templates from an ECG signal, given a list of
    R-peak locations.
    Parameters
    ----------
    signal : array
        Input ECG signal.
    rpeaks : array
        R-peak location indices.
    sampling_rate : int, float, optional
        Sampling frequency (Hz).
    before : float, optional
        Window size to include before the R peak (seconds).
    after : int, optional
        Window size to include after the R peak (seconds).
    Returns
    -------
    templates : array
        Extracted heartbeat templates.
    rpeaks : array
        Corresponding R-peak location indices of the extracted heartbeat
        templates.
    """

    # check inputs
    if signal is None:
        raise TypeError("Please specify an input signal.")

    if rpeaks is None:
        raise TypeError("Please specify the input R-peak locations.")

    if before < 0:
        raise ValueError("Please specify a non-negative 'before' value.")
    if after < 0:
        raise ValueError("Please specify a non-negative 'after' value.")

    # convert delimiters to samples
    before = int(before * sampling_rate)
    after = int(after * sampling_rate)

    # get heartbeats
    templates, newR = _extract_heartbeats(signal=signal,
                                          rpeaks=rpeaks,
                                          before=before,
                                          after=after)

    return templates

def abp_templates (wf, seg_num):
    abp=wf.segments[seg_num]['ABP']
    ecg_s=wf.chan_slice(['II'],seg_num).values.reshape(1,6400)[0]
    (ts, filtered, rpeaks, templates_ts, templates, HR_ts, HR)=ecg.ecg(ecg_s, sampling_rate=240., show=False)
    templates = extract_heartbeats(signal=abp, rpeaks=rpeaks, sampling_rate=240., before=0.0, after=0.8)
    ts = np.linspace(1,(abp.index[-1]-abp.index[0]).total_seconds()/len(templates[0]),num=len(templates[1]))

    return templates, ts

def main ():
    
    from sys import platform

    if 'linux' not in platform:
        demo_file = '/Volumes/External/Documents/Research/data/cases/Case003.hd5'
    
    else:
        demo_file = '/mnt/data01/CONDUIT/Cases/Case003.hd5'
    
    
    wf=waveform.Waveform(demo_file,start='20171213 1500',duration=600,process=True)

    #plot a good segment
    
    templates, ts = abp_templates(wf, 2)
    df = pd.DataFrame(templates).transpose() # store the templates in a single dataframe
    df.columns = df.columns.astype(str) # Bokeh wants columns to be names as str
    df['ts']=ts                         # add the time series to the df
    t_source = ColumnDataSource(df)
    
    
    p_abp_t = figure(x_axis_label='Time (s)', y_axis_label='ABP (mmHg)', 
                     tools=['box_zoom', 'xwheel_zoom', 'pan', 'reset',],
                     plot_width=400, plot_height=400, )

    p_abp_t.title.text = 'Bad Segment'
    
    for i in range(df.shape[1]-1):
        p_abp_t.line(x='ts', y=str(i), source=t_source)
    
    output_file('template.html', title='ABP Template')
    layout=column(p_abp_t)
    show(layout)
    
if __name__ == "__main__":

    main ()       