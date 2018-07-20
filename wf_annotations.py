#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 11:25:20 2018

@author: phil
"""

import pandas as pd
import sys
import glob
import os.path
#sys.path.append('/Volumes/External/Research/code/waveform_viewer/')
sys.path.append('/Volumes/External/Documents/Research/Conduit Code/')
#import waveform
#import fast_parse_xml

from biosppy.signals import ecg
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(r'/Volumes/External/Documents/Research/MATLAB');
eng.addpath(r'/Volumes/External/Documents/Research/MATLAB/WFDB');  

import numpy as np
feats_cols=['Sys_t','SBP','Dia_t','DBP','PP','MAP','Beat_P','mean_dyneg','End_sys_t','AUS','End_sys_t2','AUS2']

def annotate_file (filename, ECG_lead = 'II', R_wave = True, SQI = True, 
                   PPG_AI = True, ABP_features = True):

    # annotate hdf5 file with specified parameters 

    pass

def _read_chunk (filename, chunk_number, chunk_length = 60):

    chunksize = chunk_length * 240
    store = pd.HDFStore(filename)
    
    nrows = store.get_storer('Waveforms').nrows
#    n = chunk_number * chunksize   
    start = chunk_number * chunksize
    stop = (chunk_number + 1) * chunksize    
    
    if (stop > nrows):
        raise ('_read_chunk: Index out of range')
        store.close()
        
    else:
        chunk = store.select('Waveforms', start = start, stop = stop)
        store.close()
        return chunk
 
    
def _read_time (filename, start, duration = 60):
# read a segment of the specified file with default duration of 60 seconds
    start_time = pd.to_datetime(start)
    print ('Reading from {} for {} s'.format(start_time, duration))
    end_time = start_time + pd.to_timedelta(duration, 'S')

    df = pd.read_hdf(filename,'Waveforms', where = 'index>start_time & index<end_time')

    return df

#df2 = pd.read_hdf('Case001.hd5','Waveforms',where="index>'2017-11-14 16:46' & index<'2017-11-14 16:47'", \
#                 columns=['AR2','II','III','V','SPO2'])

def process_chunk (df):
    
    R_ts = []
    feats_cols=['Sys_t','SBP','Dia_t','DBP','PP','MAP','Beat_P','mean_dyneg','End_sys_t','AUS','End_sys_t2','AUS2']

    #biosppy for R-peaks
    ECG_sig = df['II']
    try: 
        R_peaks = np.array(ecg.christov_segmenter(signal=ECG_sig.values, sampling_rate=240.))[0]
        #            R_plot = pd.Series(index = lead1.index[R_peaks], data = lead1.values[R_peaks])
        R_ts = ECG_sig.index[R_peaks]

        # HR... 
#        mean_HR = ecg.ecg(ECG_sig.values, sampling_rate = 240., show = False)[6].mean()  
#        print ('Mean HR = {}'.format(mean_HR))
    except:
        pass  
    
    #matlab for sqi
    # need to specify which AR channel here...
    # drop NaN
    # if AR1 in columns
    # if AR2 in columns...
    df = df.dropna(axis=1,how='all')
    if 'AR1' in df.columns:
        seglist = df['AR1'].values.tolist()
#        print ('Processing AR1')
    elif 'AR2' in df.columns:
#        print ('Processing AR2')
        seglist = df['AR2'].values.tolist()
    elif 'AR3' in df.columns:
#        print ('Processing AR3')
        seglist = df['AR3'].values.tolist()
    else:
#        print ('No ABP in this chunk')
        return 0, 0, R_ts
#    print ('Calling Matlab...')    
    try:
        (onsets,feats, R, QF) = eng.wabp_wrap(seglist, nargout=4)
        feats_df = pd.DataFrame(data=np.asarray(feats),columns=feats_cols)
        MAP = feats_df['MAP'].mean()
#        print ('MAP = {0:0.2f}, SQI = {1:0.1f}'.format(MAP, QF))
    except:
#        print ('Error')
        QF = 0.0
        MAP = 0.0
    
    if  not isinstance(QF, float):
        QF = 0.0
    
    return QF, MAP, R_ts



def process_file(filename):
    
    quality = {}
    R_list = []
    store = pd.HDFStore(filename)
    nrows = store.get_storer('Waveforms').nrows
    
    chunksize = 240*60
    
    for i in range(nrows//chunksize + 1):
        chunk = store.select('Waveforms',
                                start=i*chunksize,
                                stop=(i+1)*chunksize)
#        print ('Processing chunk {} at time {}'.format(i, chunk.index[0]))
        
        if len(chunk) > 0:
        
            QF, MAP, Rs = process_chunk(chunk)
            quality[chunk.index[0]]=(MAP, QF)
            R_list.extend(list(Rs))
            print('QF = {}'.format(QF))

    store.close()    
    qdf=pd.DataFrame(index=list(quality.keys()), data=list(quality.values()),columns=['MAP','SQI'])
    qdf.index.name = 'DateTime'
    SQI = qdf['SQI']
    SQI.to_hdf(filename,'Derived/SQI',append = True, format = 'table')
    R_ts = pd.Series(data = R_list, name = 'R Peaks (raw)')
    # R_df = R_ts.to_frame()
    # R_df['Beat Type']='U'
    
    R_ts.to_hdf(filename,'/Waveform_annotations/ecg/R_peaks', append = True, format = 'table')
    
        
#    outfile = hdf_file.split('.')[0] + '.csv'
#    qdf.to_csv(outfile)



## main loop

#source = pd.read_csv('files.csv')
#source=source.dropna()


#for idx, row in source.iterrows():
#    xml_file = row['filename']
#    summary_file = row['ID'] + '.sum'
#    hdf_file = row['ID'] + '.hd5'
#    if idx > 3:
#        print ('Parsing {}'.format(hdf_file))
        
        # do SQI calculations
#        process_file (hdf_file)
    # R-R intervals

# main - process files in path specified in argv[1]

def main ():

    source = sys.argv[1]
    print ('path is {}',format(source))

    files = glob.glob(source + '/*.hd5')
#    print (files)
  
    
    for file in files:
        print('Processing {}'.format(file))
        process_file(file)

if __name__ == "__main__":

    main ()   
        
    