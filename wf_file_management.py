#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 20:46:32 2018

@author: phil

Waveform classification database management functions for waveform exploration
Workflow is based on a specific sqlite DB file for each task

Currently implemented: 
    - make file table
    - read all hdf5 files in a specified directory

To do:
    - implement read and build file table when given a csv list of files (typically this will be done once we have the entire dataset coverted
    or possibly with specific MIMIC data)

"""

import sqlite3
import os.path
import pandas as pd
import glob

def make_file_table(db_file):
    print ('Opening database connection')
    db = sqlite3.connect(db_file)
    cursor = db.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS files(filename TEXT PRIMARY KEY,
                  path TEXT,
                  start_time TEXT,
                  end_time TEXT,
                  event_time TEXT, 
                  status INTEGER)''')
    # Commit the change
    db.commit()
    db.close()
    print ('File table created')
    
def make_seg_table (db_file):
    
    db = sqlite3.connect(db_file)
    cursor = db.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS segments(id INTEGER PRIMARY KEY,
                  entry TEXT,
                  file TEXT,
                  seg INTEGER,
                  seg_start_time TEXT,
                  cA8 FLOAT,
                  cD3 FLOAT, 
                  cD4 FLOAT,
                  cD5 FLOAT,
                  cD6 FLOAT,
                  cD7 FLOAT,
                  cD8 FLOAT,
                  MAP FLOAT,
                  HR FLOAT,
                  seg_class TEXT)''')
    # Commit the change
    db.commit()
    db.close()


def build_db (db_file, source):
    # if source is a file - interpret as csv
    # if source is a directory, read all hd5 files
    db = sqlite3.connect(db_file)
    
    if os.path.isdir(source):
        print ('Reading hd5 files from the path: {}'.format(source))
        files = glob.glob(source + '/*.hd5')
        print (files)
        names = []
        start_times = []
        end_times = []
        
        for file in files:
        
            names.append(os.path.split(file)[1].split('.')[0])
            # get start and stop times
            store = pd.HDFStore(file, 'r')
            
            print('Opening file {}'.format(file))
            
            if '/Waveforms' in list(store.keys()):
                start_times.append(store.select('/Waveforms', start=1, stop=2).index[0].strftime('%Y%m%d %H:%M:%S'))
                end_times.append(store.select('/Waveforms', start=-2, stop=-1).index[0].strftime('%Y%m%d %H:%M:%S'))
            else:
                print ('Error with file {}'.format(file))
                start_times.append('BAD')
                end_times.append('BAD')
                
            store.close()
#            except:
#                print ('Error with file {}'.format(file))
        data = {'filename':names, 'path':files, 'start_time':start_times, 'end_time':end_times}
        df = pd.DataFrame(data, columns=['filename','path','start_time','end_time'])
        df['status']=0
        df = df[df.start_time != 'BAD']
        print(df)
        
        df.to_sql('files', db, if_exists='replace', index=False)
        
        # true - load all the files
        pass
    
    elif os.path.isfile(source):
        print ('Will read from the file {}'.format(source))
        # ***** TODO ****** 
        # this will read a list of files from a csv and build the files table
        # file
        pass
    else:
        raise Exception('Source is not a file or path')
    # collect files - read each one to get start and start times
    # create database

