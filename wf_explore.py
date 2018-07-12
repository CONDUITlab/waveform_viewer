#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 21:40:00 2018

Initial attempt a Bokeh based waveform visualization and annotation tool

Makes use of a sqlite DB file to define workflow for the specific analysis

    DB file can be created using wf_file_management.py function: 
        wf_file_management.build_db('workflow.db', <hdf5-directory>) 
        <hdf5-directory> is a directory containing at least 1 converted hdf5 file
    
    sqlite tables:
        files:          contains names and paths to hdf5 files to be classified
        segments:       each segment in the region of interest will be saved as a row along with wavelet coefficients and other hemodynamic parameters 
        segment_types:  (not yet implemented) but this will define the possible segment classifications (currently defined as wf_types )

    Major Dependencies:
        biosppy library:    ecg analysis - primarily used for HR but also R-peak detection
        waveform.py:        contains the classes for waveform, summary and wavelet objects
        wabp_wrap.m:        matlab wrapper for physionet functions.. the path is (badly) hard coded in waveform.py
        wfdb matlab fns:    these provide values for MAP and signal quality index (SQI) for each segment
        
        Note: waveform.py makes use of the matlab engine which is a pain to use... see the matlab docs but you will 
    
    
    Usage (local machine): hil$ bokeh serve --show wf_explore.py --args 'workflow.db' 

    Global to do:
        1. clean up messy code!
        2. implement for remote server (local browser)
        3. implement wavelet heatmap with linked waveform view
        4. implement annotation view

"""

import numpy as np
import pandas as pd
import waveform
import sqlite3
import os.path
import itertools

from math import pi
from bokeh.models import DatetimeTickFormatter

from bokeh.plotting import figure 
from bokeh.palettes import Category10
from bokeh.io import output_notebook, show, output_file, curdoc
from bokeh.layouts import column, row, layout
from bokeh.models import ColumnDataSource, HoverTool, RadioButtonGroup, Span
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Slider, RangeSlider, CheckboxGroup,  Button, TextInput, Paragraph
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
import sys

hover = HoverTool(
    tooltips=[
        ('Time','@index{%H:%M:%S:%3Nms}'),
        ('index','$index')
    ],
    mode='vline'
    )
    
hover.formatters = {'index':'datetime'}

################################## Miscellaneous functions ##################################

def duration_HMS(start, stop):
    duration = (stop-start).total_seconds()
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod (remainder, 60)
    return hours, minutes, seconds

def color_gen():
    for arg in itertools.cycle(Category10[10]):
        yield arg 
colors = color_gen()   

################################## Initialization ##################################

# Reads input .hdf files from .db file
def read_files (db_file):
   db = sqlite3.connect(db_file)
   df = pd.read_sql('select * from files',db)
   db.close()
   return df
   
# open database file
db_file = sys.argv[1] # db file is the master file for the workflow (contains files and classified segments)
print ('Opening workflow file/database in {}'.format(db_file))
files = read_files(db_file) # consider reading only files with specific status or filter the table (eg hide files that are already completed)

# open waveform file - this should be done in the file_management tab
cur_file_name = files.filename[0] # the current file should be the file selected from the files table in db_file
active_file = files.path[0]

################################## File Management ##################################

## File Management callbacks ##

def file_select(attr, old, new):
    global active_file
   
    try:
        selected_index = file_table.selected["1d"]["indices"][0]
        cur_file_name = file_table.data['filename'][selected_index]
        active_file = file_table.data['path'][selected_index]
        sel_file_txt.text = 'Current File: '+ str(active_file.split('\\')[-1])
        cur_file_box.text = 'Current File: '+ str(active_file.split('\\')[-1])
        selected_file.text = 'Current File: '+ str(active_file.split('\\')[-1])
        # update vitals_plot
        vs_sum = waveform.Summary(active_file)
        vs_source.data = ColumnDataSource(data=vs_sum.data).data
        p_ABP.title.text = 'Vitals Summary for file: {}'.format(cur_file_name )
        
        #show in terminal window
        print('New file selected: {}'.format(cur_file))
    except IndexError:
        pass

## File management widgets ##
sel_file_txt = Paragraph(text='Current File: '+ str(active_file.split('\\')[-1]))

files['start_time'] = pd.to_datetime(files['start_time'])
file_table = ColumnDataSource(files)

columns = files.columns
columns = [
    TableColumn(field='filename', title='File'),
    TableColumn(field='start_time', title='Start Time', formatter=DateFormatter()),
    TableColumn(field='status', title='status'),
]

data_table = DataTable(source = file_table, columns=columns, width=400, height=280)

## Add callbacks to widgets ##
file_table.on_change('selected', file_select)

table_widget = widgetbox(data_table)
file_controls = widgetbox(sel_file_txt)
file_layout = column(table_widget, file_controls)
file_tab = Panel(child=file_layout, title='File Management')

##################################  Vitals Panel ##################################

def createVitalsPanel():
    colors = color_gen()
    for elem, color in zip(list(vs_sum.data), colors):
        if elem == 'AR2-M':
            continue
        p_temp = figure(y_axis_label=elem, x_axis_type='datetime', plot_width=vs_width, plot_height=vs_height)
        p_temp.xaxis.formatter=DatetimeTickFormatter(
            hours=dt_axis_format,
            days=dt_axis_format,
            months=dt_axis_format,
            years=dt_axis_format,
        )
        p_temp.x_range = p_ABP.x_range
        p_temp.line('DateTime', elem, source=vs_source, line_color=color)
        p_dict[elem] = p_temp
    selected_dates.text=str(min(vs_sum.data.index)) + ' to ' + str(max(vs_sum.data.index))
    checkbox_group.labels = list(p_dict)
    checkbox_group.active = list(range(0,len(p_dict)))
    s = min(vs_sum.data.index).timestamp()*1000
    e = max(vs_sum.data.index).timestamp()*1000
    date_range_slider.start = s
    date_range_slider.end = e
    date_range_slider.value = (s, e)
    vs_layout.children = None
    vs_layout.children.append(widgetbox(selected_file, selected_duration, load_button, checkbox_group, date_range_slider, selected_dates))
    vs_layout.children.append(p_ABP)
    for elem in p_dict:
        vs_layout.children.append(p_dict[elem])

## Vitals Functions ##

vs_sum = waveform.Summary(active_file) # Get summary of vitals from active file
vs_source = ColumnDataSource (data=vs_sum.data)

vs_hover = HoverTool(
    tooltips=[
        ('Time','@DateTime{%Y-%m-%d-%H:%M}'),
        ('index','$index')
    ],
    mode='vline'
    )
vs_hover.formatters = {'DateTime':'datetime'}

vs_width = 1200
vs_height = 250

# Plot ABP (Change this so that it can be changed to CVP etc.) 
p_ABP = figure(y_axis_label='ABP (mmHg)', x_axis_type='datetime', 
           tools=['box_zoom', 'xwheel_zoom', 'pan', vs_hover, 'reset','crosshair'], y_range=(0, 200), 
               plot_width=vs_width, plot_height=vs_height, title = 'Vitals Summary for file: {}'.format(cur_file_name ))
p_ABP.title.align = 'center'
dt_axis_format = ["%d-%m-%Y %H:%M"]

p_ABP.xaxis.formatter=DatetimeTickFormatter(
        hours=dt_axis_format,
        days=dt_axis_format,
        months=dt_axis_format,
        years=dt_axis_format,
    )
p_ABP.line('DateTime', 'AR2-M', source=vs_source, line_color = next(colors), legend='AR2')

# Start span represents the beginning of the area of interest
start_span = Span(location=min(vs_sum.data.index).timestamp()*1000,
                  dimension='height', line_color='green',
                  line_dash='dashed', line_width=3)
p_ABP.add_layout(start_span)

# End span represents the end of the area of interest
end_span = Span(location=max(vs_sum.data.index).timestamp()*1000,
                  dimension='height', line_color='red',
                  line_dash='dashed', line_width=3)
p_ABP.add_layout(end_span)
    
p_dict = {}
for elem in list(vs_sum.data):
    if elem == 'AR2-M':
        continue
    p_temp = figure(y_axis_label=elem, x_axis_type='datetime', plot_width=vs_width, plot_height=vs_height)
    p_temp.xaxis.formatter=DatetimeTickFormatter(
        hours=dt_axis_format,
        days=dt_axis_format,
        months=dt_axis_format,
        years=dt_axis_format,
    )
    p_temp.x_range = p_ABP.x_range
    p_temp.line('DateTime', elem, source=vs_source, line_color=next(colors))
    p_dict[elem] = p_temp

load_button = Button(label="Segment File", button_type="success")
selected_file = Paragraph(text = 'Current File: {0:s}'.format(os.path.split(active_file)[-1]))

hours, minutes, seconds = duration_HMS(min(vs_sum.data.index), max(vs_sum.data.index))
selected_duration = Paragraph(text = 'Selected Duration: {0:0.0f}:{1:02.0f}:{2:02.0f}'.format(hours, minutes, seconds))

selected_dates = Paragraph(text=str(min(vs_sum.data.index).round('s')) + ' to ' + str(max(vs_sum.data.index).round('s')))
checkbox_group = CheckboxGroup(labels = list(p_dict), active = list(range(0,len(p_dict))), inline = True)

s = min(vs_sum.data.index).timestamp()*1000
e = max(vs_sum.data.index).timestamp()*1000
date_range_slider = RangeSlider(title="Date Range", start=s, end=e, value= (s, e), step=1, show_value = False, tooltips = False)

def load_cb ():
    global wf
    global wvt
    load_button.disabled = True
    load_button.label = 'Segmenting File'
    load_button.button_type = 'warning'
    seg_slider.disabled = True
    save_seg_button.disabled = True
    dates = date_range_slider.value
    vs_start = pd.Timestamp(dates[0]/1000,unit='s')
    vs_end = pd.Timestamp(dates[1]/1000,unit='s')
    
    print ('Current x range is from {} to {})'.format(vs_start.round('s'), vs_end.round('s')))
    
    start_str = vs_start.strftime('%Y%m%d-%H%M%S')
    end_str = vs_end.strftime('%Y%m%d-%H%M%S')
    wf=waveform.Waveform(active_file, start=start_str, end=end_str, process=True )
    wvt = waveform.ABPWavelet(wf, process=True)
    print ('Read complete')
    
    seg_slider.value = 1
    seg_slider.end=len(wf.segments)
    wf_source.data = ColumnDataSource(wf.segments[1].to_frame()).data
    p.line('index','ABP',source=wf_source, name = 'segment_line')
    
    load_button.disabled = False
    save_seg_button.disabled = False
    seg_slider.disabled = False
    load_button.label = 'Segment File'
    load_button.button_type = 'success'
    
    
def checkbox_click_handler(selected_checkboxes):
    visible_glyphs = vs_layout.children
    for index, glyph in enumerate(p_dict):
        if index in selected_checkboxes:
            if p_dict[glyph] not in visible_glyphs:
                vs_layout.children.append(p_dict[glyph])
        else:
            if p_dict[glyph] in visible_glyphs:
                vs_layout.children.remove(p_dict[glyph])
                
def date_time_slider(attr, old, new):
    dates = date_range_slider.value
    vs_start = pd.Timestamp(dates[0]/1000,unit='s')
    vs_end = pd.Timestamp(dates[1]/1000,unit='s')
    hours, minutes, seconds = duration_HMS(vs_start, vs_end)
    fname = os.path.split(active_file)[1]
    print('Current File: {0:s}\nSelected Duration: {1:0.0f}:{2:02.0f}:{3:02.0f}'.format(fname, hours,minutes,seconds))
    selected_duration.text = ('Selected Duration: {0:0.0f}:{1:02.0f}:{2:02.0f}'.format(hours,minutes,seconds))
    selected_dates.text =str(vs_start.round('s')) + ' to ' + str(vs_end.round('s'))
    start_span.location = dates[0]
    end_span.location = dates[1]


load_button.on_click(load_cb)
checkbox_group.on_click(checkbox_click_handler)
date_range_slider.on_change('value',date_time_slider)

vs_layout = column()
vs_layout.children.append(widgetbox(selected_file, selected_duration, load_button, checkbox_group, date_range_slider, selected_dates))
vs_layout.children.append(p_ABP)
for elem in p_dict:
    vs_layout.children.append(p_dict[elem])
    
vs_tab = Panel(child=vs_layout, title = 'Vitals')

##################################  Waveform Panel ##################################

## Waveform callbacks ##

def save_button_cb ():
    choice = wf_types[rbg.active]
    N = seg_slider.value
    print ('Segment {} annotated as {}'.format(N,choice))
    entry = cur_file_name + '_{0:03d}'.format(N)
    db_row = wvt.wltFeatures.loc[N:N].copy()
    db_row['seg'] = N
    db_row['file'] = cur_file_name
    db_row['entry'] = entry
    db_row['seg_class'] = choice # assign the classification to the entry row
    db_row['seg_start_time'] = wvt.seg_start_time[N]
    # write to the database
    db = sqlite3.connect(db_file)
    db_row.to_sql("segments", db, if_exists='append', index=False)
    db.close()
    print (db_row.head(1))
    seg_slider.value += 1

def slider_plus(): 
    if seg_slider.value != seg_slider.end:
        seg_slider.value += 1
    
def slider_minus(): 
    if seg_slider.value != seg_slider.start:
        seg_slider.value -= 1
    
    
def callback (attr, old, new):
    # segment selection callback
    # add functionality to update the segment classification selector based on previously assigned classification (eg rbg.active)
    
    N = seg_slider.value
#    print(N)
    try: wf_source.data = ColumnDataSource(wf.segments[N].to_frame()).data
    except NameError: print('wf not defined yet. Sldier change is null')


cur_file_box = Paragraph(text='Current File: '+ str(active_file.split('\\')[-1]))

# initially fill source with a blank dataframe and update it when then region of interest is selected in the vitals tab
wf_source=ColumnDataSource()

p = figure(x_axis_label='Datetime',y_axis_label='ABP (mmHg)', x_axis_type='datetime', 
          tools=['box_zoom', 'xwheel_zoom', 'pan', hover, 'reset','crosshair'], plot_width=1000)


seg_slider = Slider(start=1, end=2, value=1, step=1, title="Segment", disabled = True)
save_seg_button = Button(label='Save Segment', button_type='success', disabled = True)
    
seg_slider.on_change('value', callback)    
save_seg_button.on_click(save_button_cb)
    
wf_types = ['Unclassified', 'Normal', 'Abnormal']  # read this from the workflow file as well - table: waveform_types

rbg = RadioButtonGroup ( labels = wf_types, active = 0)
plus = Button(label = '+')
minus = Button(label = '-')
plus.on_click(slider_plus)
minus.on_click(slider_minus)

wf_layout = layout(children = [cur_file_box, row(minus, seg_slider, plus), rbg, save_seg_button])
#wf_layout.children.append(column(cur_file_box, rbg, save_seg_button))
wf_layout.children.append(p)
wf_tab = Panel(child = wf_layout, title = 'Waveforms')

##################################  Bokeh Output ##################################

# combine the panels and plot
layout = Tabs(tabs=[ file_tab, vs_tab, wf_tab])

curdoc().add_root(layout)

# If not being executed on bokeh server, these will provide output for user
output_file('wf_view.html')
show (layout)


