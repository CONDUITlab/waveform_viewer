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
        segment_types:  (not yet implemented) but this will define the possible segment classifications (currently defined as wf_classes )

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

import sqlite3
import os.path
import itertools

from math import pi
import matlab.engine
from bokeh.models import DatetimeTickFormatter, PointDrawTool

from bokeh.plotting import figure 
from bokeh.palettes import Category10
from bokeh.io import output_notebook, show, output_file, curdoc
from bokeh.layouts import column, row, layout
from bokeh.models import ColumnDataSource, HoverTool, RadioButtonGroup, Span, LinearAxis, Range1d
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Slider, RangeSlider, CheckboxGroup,  Button, TextInput, Paragraph
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
import sys
import waveform

from participant import participant
from xlrd import open_workbook
import os
import glob

# Hover tool definitions
hover = HoverTool(
    tooltips=[
        ('Time','@index{%H:%M:%S:%3Nms}'),
        ('index','$index')
    ],
    mode='vline'
    )
hover.formatters = {'index':'datetime'}

vs_hover = HoverTool(
    tooltips=[
        ('Time','@DateTime{%Y-%m-%d-%H:%M}'),
        ('index','$index')
    ],
    mode='vline'
    )
vs_hover.formatters = {'DateTime':'datetime'}

dt_axis_format = ["%d-%m-%Y %H:%M"]
vs_x_axis = DatetimeTickFormatter(
            hours=dt_axis_format,
            days=dt_axis_format,
            months=dt_axis_format,
            years=dt_axis_format,
        )

wf_classes = ['Unclassified', 'Normal', 'Abnormal']  # read this from the workflow file as well - table: waveform_types

vs_types = ['AR1-M','AR2-M','AR3-M','CVP1','CVP2']
wf_present = {x:True for x in vs_types}
wf_names = [['AR1-M','DateTime'],['AR2-M','DateTime'],['AR3-M','DateTime'],['CVP1','DateTime'],['CVP2','DateTime']]
wf_radio_button = RadioButtonGroup ( labels = vs_types, active = 0)

wf_types = ['AR1','AR2','AR3','CVP1','CVP2']
visual_vitals = ['HR','SPO2-%','PVC','NBP-M']

# Change path depending on server use 
if 'linux' in sys.platform:
    base_path = '/mnt/'
else:
    base_path = 'Y:/'

dir_in_str = os.fsencode(base_path + 'data03/workspace/pressorsHRV/data/patient_files/')
dir_waveform_str = os.fsencode(base_path + 'data03/workspace/pressorsHRV/csvCombined/BP/')
dir_HRV_str = os.fsencode(base_path + 'data03/workspace/pressorsHRV/csvCombined/new/')
dicts = []

################################## Miscellaneous functions ##################################

# Returns the time difference between two datetimes in hours, minutes and seconds respectively
def duration_HMS(start, stop):
    duration = (stop-start).total_seconds()
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod (remainder, 60)
    return hours, minutes, seconds

#Bokeh color iterator
def color_gen():
    for arg in itertools.cycle(Category10[10]):
        yield arg 
colors = color_gen()   

################################## Initialization ##################################

# Reads input .hdf files from .db file
def read_files (db_file):
    if not os.path.isfile(db_file):
        raise Exception('.db file does not exist')
    db = sqlite3.connect(db_file)
    df = pd.read_sql('select * from files',db)
    db.close()
    return df

# Attempts to find HRV data based on file name    
def getHRV():
    found = False
    for file in os.listdir(dir_in_str):
        if cur_file_name.split('_')[0] == file.replace(b' ',b'').split(b'-')[0].decode("utf-8"):
            filename = os.fsdecode(file)
            if filename.endswith('.xlsx'): #open excel files
                fn = filename.replace(" ","")
                base = fn.split(".")[0]
                bbase = base.split("-")[0]
                print('Found pressor file: ' + bbase)
                #load excel patient data
                wb = open_workbook(dir_in_str.decode("utf-8")+file.decode("utf-8"))
                p = participant(wb)
                #create the data frames
                p.create_dfs_all()
                #get waveform files
                HRVfiles = glob.glob(dir_waveform_str.decode("utf-8")+bbase+'_MAP.*')
                if len(HRVfiles) > 0:
                    p.add_AdditionalWaveforms(HRVfiles[0])
                else:
                    p.DF_Waveforms = None
                #get HRV files
                HRVfiles = glob.glob(dir_HRV_str.decode("utf-8")+bbase+'_HRV.*')
                if len(HRVfiles) > 0:
                    p.add_HRV(HRVfiles[0])
                else:
                    p.DF_HRV = None
            found = True
            break
    if not found:
        print('No pressor file found')
        return None
    # Convert the pressor data into a format more suitable for plotting
    p.DFpressors = p.DFpressors.drop(['study_id'],axis='columns')
    p.DFpressors['DateTime'] = p.DFpressors.index
    p.DFpressors = p.DFpressors.melt(id_vars=['DateTime']).dropna(axis='rows',how='any').set_index('DateTime')
    pressor_source.data = ColumnDataSource(p.DFpressors).data # updates pressor plot source
    return p
   
# open database file
db_file = sys.argv[1] # db file is the master file for the workflow (contains files and classified segments)
print ('Opening workflow file/database in {}'.format(db_file))
files = read_files(db_file) # consider reading only files with specific status or filter the table (eg hide files that are already completed)
vs_sum = None
# open waveform file - this should be done in the file_management tab
for selected_index in range(0, len(files)):
    cur_file_name = files.filename[selected_index] # the current file should be the file selected from the files table in db_file
    active_file = files.path[selected_index]
    vs_sum = waveform.Summary(active_file) # Get summary of vitals from active file
    if not set(wf_names[wf_radio_button.active]).isdisjoint(list(vs_sum.data)):
        break
# If the selected waveform cannot be found raise an error
if set(wf_names[wf_radio_button.active]).isdisjoint(list(vs_sum.data)):  
    raise('No ' + vs_types[wf_radio_button.active] + 'signal')
pressor_source = ColumnDataSource()
p = getHRV()

################################## File Management ##################################

## File Management callbacks ##
def update():
    global active_file, selected_index, vs_source
    # Get the row number of the selected file
    selected_index_temp = file_table.selected["1d"]["indices"][0]
    # Change warning to match selection
    warn_txt.text = 'No ' + vs_types[wf_radio_button.active] + ' found in selection'
    # Check that the selected file has the waveform of interest
    if set(wf_names[wf_radio_button.active]).isdisjoint(list(waveform.Summary(file_table.data['path'][selected_index_temp]).data)):
        # If it isn't there, ignore the change
        if warn_txt not in file_layout.children:
            file_layout.children.append(warn_txt)
        file_table.selected.indices = [selected_index]
        return
    # Remove the 'missing' warning
    if warn_txt in file_layout.children:
        file_layout.children.remove(warn_txt)
    # If the same file was selected, ignore the change
    if selected_index == selected_index_temp:
        return
    # Disable the waveform tab (so the .db file isn't overwritten with a different file)
    disable_wf_panel()
    # Change selection
    selected_index = selected_index_temp
    cur_file_name = file_table.data['filename'][selected_index]
    active_file = file_table.data['path'][selected_index]
    p = getHRV()
    sel_file_txt.text = 'Current File: '+ str(active_file.split('\\')[-1])
    cur_file_box.text = 'Current File: '+ str(active_file.split('\\')[-1])
    selected_file.text = 'Current File: '+ str(active_file.split('\\')[-1])
    # Update the source of the plots
    vs_sum = waveform.Summary(active_file)
   
    # Reset vitals selections
    for elem in vs_types:
        if elem not in list(vs_sum.data): 
            vs_sum.data[elem] = [np.nan] * len(vs_sum.data.index)
            wf_present[elem] = False
        else: wf_present[elem] = True 
    for elem in visual_vitals:
        if elem not in list(vs_sum.data): 
            vs_sum.data[elem] = [np.nan] * len(vs_sum.data.index)
    # Update plot data
    vs_source.data = ColumnDataSource(data=vs_sum.data).data
    date_range_slider.start = pd.to_datetime(min(vs_sum.data.index)).timestamp()*1000
    date_range_slider.end = pd.to_datetime(max(vs_sum.data.index)).timestamp()*1000
    date_range_slider.value = (date_range_slider.start, date_range_slider.end)
    p_main.title.text = 'Vitals Summary for file: {}'.format(cur_file_name )
    p_main.x_range.start = date_range_slider.start
    p_main.x_range.end = date_range_slider.end
    
    print('New file selected: {}'.format(cur_file_name))
    
## File management widgets ##
sel_file_txt = Paragraph(text='Current File: '+ str(active_file.split('\\')[-1]))
warn_txt = Paragraph(text='No ' + vs_types[wf_radio_button.active] + ' found in selection')
sel_file_button = Button(label='Change File',button_type='success')

files['start_time'] = pd.to_datetime(files['start_time'])
file_table = ColumnDataSource(files)

columns = [
    TableColumn(field='filename', title='File'),
    TableColumn(field='start_time', title='Start Time', formatter=DateFormatter()),
    TableColumn(field='status', title='Status'),
]

data_table = DataTable(source = file_table, columns=columns, width=400, height=280)

## Add callbacks to widgets ##
file_table.selected.indices = [selected_index]
sel_file_button.on_click(update)

table_widget = widgetbox(data_table)
file_controls = row(sel_file_txt,sel_file_button)
file_layout = column(table_widget, file_controls)
file_tab = Panel(child=file_layout, title='File Management')

##################################  Vitals Panel ##################################

## Vitals Functions ##

# Fill the vitals with NaN values if not present so that the plots can update with other files that
#   do contain these columns
for elem in vs_types:
    if elem not in list(vs_sum.data): 
        vs_sum.data[elem] = [np.nan] * len(vs_sum.data.index)
        wf_present[elem] = False
    else: wf_present[elem] = True       

vs_width = 1200
vs_height = 250

vs_source = ColumnDataSource (data=vs_sum.data)

# Create main figure (for ABP / CVP)
p_main = figure(y_axis_label='ABP (mmHg)', x_axis_type='datetime', 
           tools=['box_zoom', 'wheel_zoom', 'pan', vs_hover, 'reset','crosshair'], y_range=(0, 200), 
               plot_width=vs_width, plot_height=int(vs_height*1.3), title = 'ABP Summary for file: {}'.format(cur_file_name ))
p_main.title.align = 'center'

p_main.xaxis.formatter = vs_x_axis
# Plot the vitals on the main plot
r_main = [p_main.line('DateTime', elem, source=vs_source, line_color = next(colors)) for elem in vs_types]

# Change all lines to be invisible
for elem in r_main: elem.glyph.line_alpha = 0
# Change all active lines to be visible
r_main[wf_radio_button.active].glyph.line_alpha = 1
  
# Start span represents the beginning of the area of interest
start_span = Span(location=min(vs_sum.data.index).timestamp()*1000,
                  dimension='height', line_color='green',
                  line_dash='dashed', line_width=3)
p_main.add_layout(start_span)

# End span represents the end of the area of interest
end_span = Span(location=max(vs_sum.data.index).timestamp()*1000,
                  dimension='height', line_color='red',
                  line_dash='dashed', line_width=3)
p_main.add_layout(end_span)

# Setting the second y axis range name and range
p_main.extra_y_ranges = {"pressor": Range1d(start=0, end=20)}

# Adding the second axis to the plot.  
p_main.add_layout(LinearAxis(y_range_name="pressor"), 'right')

# Add pressor data to main figure if the data exists
if p:
    # Pressor data has a y value of the 'amount' 
    pressor_glyph = p_main.circle(x = 'DateTime',y='value',source = pressor_source,color = next(colors),y_range_name="pressor")
    PressorHoverTool = HoverTool(
        name= 'Pressor Hover',
        renderers = [pressor_glyph],
        tooltips=[
            ( 'Pressor',   '@variable'            ),
            ( 'Amount',  '@value' ), # use @{ } for field names with spaces
        ]
    )
    p_main.add_tools(PressorHoverTool)
         
p_dict = {}
# Plot all other existing vitals
for elem in visual_vitals:
    if elem not in list(vs_sum.data): 
        vs_sum.data[elem] = [np.nan] * len(vs_sum.data.index)
    p_dict[elem] = figure(y_axis_label=elem, x_axis_type='datetime', plot_width=vs_width, plot_height=int(vs_height*1.3),tools=['box_zoom', 'xwheel_zoom', 'pan'])
    p_dict[elem].xaxis.formatter = vs_x_axis
    p_dict[elem].x_range = p_main.x_range
    p_dict[elem].line('DateTime', elem, source=vs_source, line_color=next(colors))

seg_button = Button(label="Segment File", button_type="success")
selected_file = Paragraph(text = 'Current File: {0:s}'.format(os.path.split(active_file)[-1]))

# Get the start and end datetime values of the data
vs_start = min(vs_sum.data.index)
vs_end = max(vs_sum.data.index)

# Print the total time difference between the start and end
hours, minutes, seconds = duration_HMS(vs_start, vs_end)
selected_duration = Paragraph(text = 'Selected Duration: {0:0.0f}:{1:02.0f}:{2:02.0f}'.format(hours, minutes, seconds))

selected_dates = Paragraph(text=str(vs_start.round('s')) + ' to ' + str(vs_end.round('s')))
checkbox_group = CheckboxGroup(labels = list(p_dict), active = list(range(0,len(p_dict))), inline = True)

date_range_slider = RangeSlider(title="Date Range", start=vs_start.timestamp()*1000, end=vs_end.timestamp()*1000, value= (vs_start.timestamp()*1000, vs_end.timestamp()*1000), step=1, show_value = False, tooltips = False)

# Callback for switching the main vital display (using radiobutton group)
def wf_switch(attr, old, new):
    if wf_present[vs_types[new]]:
        p_main.yaxis.axis_label = vs_types[new]
        p_main.title.text = vs_types[new] + ' Summary for file: {}'.format(cur_file_name )
        for r in r_main:
            r.glyph.line_alpha = 0
        r_main[new].glyph.line_alpha =1
    else:
        print(vs_types[new] + ' not in selected file')
        wf_radio_button.active = old 

def load_cb ():
    global wf, wvt
    # Disable buttons while segmenting
    seg_button.disabled = True
    seg_button.label = 'Segmenting File'
    seg_button.button_type = 'warning'
    disable_wf_panel()
    # Get the dates selected on the slider and convert them to timestamps
    dates = date_range_slider.value
    vs_start = pd.Timestamp(dates[0]/1000,unit='s')
    vs_end = pd.Timestamp(dates[1]/1000,unit='s')
    
    print ('Current x range is from {} to {})'.format(vs_start.round('s'), vs_end.round('s')))
    
    start_str = vs_start.strftime('%Y%m%d-%H%M%S')
    end_str = vs_end.strftime('%Y%m%d-%H%M%S')
    
    # Perform the segmentation and wavelet operations on the selected data
    if 'AR' in vs_types[wf_radio_button.active]:
        wf = waveform.Waveform(active_file, start=start_str, end=end_str, process=True ,seg_channel = vs_types[wf_radio_button.active])
        wvt = waveform.ABPWavelet(wf, process=True)
    elif 'CVP' in vs_types[wf_radio_button.active]:
        wf = waveform.CVPWaveform(active_file, start=start_str, end=end_str, process=True ,seg_channel = vs_types[wf_radio_button.active])
        wvt = waveform.CVPWavelet(wf, process=True)
        
    # Update the segment slider
    seg_slider.value = 1
    seg_slider.end=len(wf.segments)
    
    # Update the waveform panel plots
    wf_source.data = ColumnDataSource(wf.segments[1]).data
    wf_start = pd.to_datetime(min(wf_source.data['index'])).timestamp()*1000
    wf_end = pd.to_datetime(max(wf_source.data['index'])).timestamp()*1000
    
    p_seg.yaxis.axis_label = wf_types[wf_radio_button.active]
    p_seg.x_range.start = wf_start
    p_seg.x_range.end = wf_end
    
    # Update pressor information if it exists
    if p:
        p_subset = p.DFpressors.loc[(p.DFpressors.index >= vs_start) & (p.DFpressors.index <= vs_end)]
        for elem in p_subset.index:
            print(int((elem - vs_start) / ((vs_end - vs_start) / len(wf.segments))+1))
        pressor_seg.data = ColumnDataSource(p.DFpressors.loc[(p.DFpressors.index >= pd.to_datetime(p_seg.x_range.start/1000, unit = 's')) & (p.DFpressors.index <= pd.to_datetime(p_seg.x_range.end/1000, unit = 's'))]).data
    
    wf_line.glyph.y = wf_types[wf_radio_button.active]
    p_wf_II.x_range.start = wf_start
    p_wf_II.x_range.end = wf_end
    
    # Show R peaks if selected
    
    if show_peaks.active == 'no': #deactivated
        df = pd.DataFrame(wf_source.data)
        df['DateTime'] = wf_source.data['index']
        try: ind = [x.item() for x in pd.to_numeric(df['index'])]
        except AttributeError: ind = list(pd.to_numeric(df['index']))
        
        try: ecg = [x.item()*1000 for x in df['II']]
        except AttributeError: ecg = list(df['II']*1000)
        ann, anntype = eng.wrapper(ind,ecg,'wf_files/'+active_file.split('\\')[-1].split('.')[0],240,nargout=2)
        R_peaks = [int(ann[i][0]) for i, e in enumerate(anntype) if e == 'N']
        ann_source.data = ColumnDataSource(df.iloc[R_peaks,:]).data 
        
    # Enable elements on vitals panel
    seg_button.disabled = False
    disable_wf_panel(False)
    seg_button.label = 'Segment File'
    seg_button.button_type = 'success'
    print ('Read complete')
    
# Handler for removing / adding vital plots using checkbox_group    
def checkbox_click_handler(selected_checkboxes):
    visible_glyphs = vs_plots.children
    for index, glyph in enumerate(p_dict):
        if index in selected_checkboxes:
            if p_dict[glyph] not in visible_glyphs:
                vs_plots.children.append(p_dict[glyph])
        else:
            if p_dict[glyph] in visible_glyphs:
                vs_plots.children.remove(p_dict[glyph])
                
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

seg_button.on_click(load_cb)
checkbox_group.on_click(checkbox_click_handler)
date_range_slider.on_change('value',date_time_slider)
wf_radio_button.on_change('active',wf_switch)

vs_layout = column()
vs_layout.children.append(widgetbox(wf_radio_button, selected_file, selected_duration, seg_button, checkbox_group, date_range_slider, selected_dates))

vs_layout.children.append(p_main)
vs_plots = column()

for elem in p_dict:
    vs_plots.children.append(p_dict[elem])

vs_layout.children.append(vs_plots)  
 
vs_tab = Panel(child=vs_layout, title = 'Vitals')
  
##################################  Waveform Panel ##################################

## Waveform callbacks ##

def save_button_cb ():
    choice = wf_classes[rbg.active]
    N = seg_slider.value
    print ('Segment {} annotated as {}'.format(N,choice))
#    entry = cur_file_name + '_{0:03d}'.format(N)
    entry = os.path.basename(active_file).split('.')[0].split('_case_')[1] + '_{0:03d}'.format(N)
    
    db_row = wvt.wltFeatures.loc[N:N].copy()
    db_row['seg'] = N
    db_row['file'] = active_file
    db_row['entry'] = entry
    db_row['seg_class'] = choice # assign the classification to the entry row
    db_row['seg_start_time'] = wvt.seg_start_time[N]
    db_row['seg_length']  = wvt.section_size
    db_row['channel']  = wvt.seg_channel
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
    
def disable_wf_panel(disable = True):
    seg_slider.disabled = disable
    save_seg_button.disabled = disable
    plus.disabled = disable
    minus.disabled = disable
    
def seg_callback (attr, old, new):
    # segment selection callback
    # add functionality to update the segment classification selector based on previously assigned classification (eg rbg.active)
    
    N = seg_slider.value
    wf_source.data = ColumnDataSource(wf.segments[N]).data
    if p:
        pressor_seg.data = ColumnDataSource(p.DFpressors.loc[(p.DFpressors.index >= pd.to_datetime(p_seg.x_range.start/1000, unit = 's')) & (p.DFpressors.index <= pd.to_datetime(p_seg.x_range.end/1000, unit = 's'))]).data
    if show_peaks.active == 'no':
        df = pd.DataFrame(wf_source.data)
        df['DateTime'] = wf_source.data['index']
        try: ind = [x.item() for x in pd.to_numeric(df['index'])]
        except AttributeError: ind = list(pd.to_numeric(df['index']))
        try: ecg = [x.item()*1000 for x in df['II']]
        except AttributeError: ecg = list(df['II']*1000)
        ann, anntype = eng.wrapper(ind,ecg,'wf_files/'+active_file.split('\\')[-1].split('.')[0],240,nargout=2)
        R_peaks = [int(ann[i][0]) for i, e in enumerate(anntype) if e == 'N']
        ann_source.data = ColumnDataSource(df.iloc[R_peaks,:]).data
        
    wf_start = pd.to_datetime(min(wf_source.data['index'])).timestamp()*1000
    wf_end = pd.to_datetime(max(wf_source.data['index'])).timestamp()*1000
    p_seg.x_range.start = wf_start
    p_seg.x_range.end = wf_end
    p_wf_II.x_range.start = wf_start
    p_wf_II.x_range.end = wf_end
    

cur_file_box = Paragraph(text='Current File: '+ str(active_file.split('\\')[-1]))

wf_source = ColumnDataSource(pd.read_hdf(active_file,key='Waveforms',stop=1))
df = pd.DataFrame(wf_source.data)
df['DateTime'] = wf_source.data['index']
ann_source = ColumnDataSource(df)

p_seg = figure(x_axis_label='Datetime',y_axis_label='ABP (mmHg)', x_axis_type='datetime', 
          tools=['box_zoom', 'xwheel_zoom', 'pan', hover, 'reset','crosshair'], plot_width=1000, plot_height = 400)

p_wf_II = figure(x_axis_label='Datetime',y_axis_label='II (mV)', x_axis_type='datetime', 
          tools=['box_zoom', 'xwheel_zoom', 'pan', hover, 'reset','crosshair'], plot_width=1000, plot_height = 400)
p_wf_II.x_range = p_seg.x_range

wf_line = p_seg.line('index',wf_types[wf_radio_button.active], source=wf_source, color = next(colors))

p_seg.extra_y_ranges = {"pressor_wf": Range1d(start=0, end=20)}
p_seg.add_layout(LinearAxis(y_range_name="pressor_wf"), 'right')
if p:
    pressor_seg = ColumnDataSource(p.DFpressors)
    pressor_seg_glyph = p_seg.circle(x = 'DateTime',y='value',source = pressor_seg,color = 'red',y_range_name="pressor_wf")
    PressorSegHoverTool = HoverTool(
        name= 'Pressor Hover',
        renderers = [pressor_seg_glyph],
        tooltips=[
            ( 'Pressor',   '@variable'),
            ( 'Amount',  '@value' ),
        ]
    )
    p_seg.add_tools(PressorSegHoverTool)
p_wf_II.line('index','II', source=wf_source, color = next(colors))
II_c = p_wf_II.circle(x='DateTime',y='II',source=ann_source,color=next(colors))
point_draw = PointDrawTool(renderers=[II_c])
p_wf_II.add_tools(point_draw)

seg_slider = Slider(start=1, end=2, value=1, step=1, title="Segment", disabled = True)
save_seg_button = Button(label='Save Segment', button_type='success', disabled = True)
    
seg_slider.on_change('value', seg_callback)    
save_seg_button.on_click(save_button_cb)
show_peaks = CheckboxGroup(labels = ['R Peaks'], active = [0])
rbg = RadioButtonGroup ( labels = wf_classes, active = 0)
plus = Button(label = '+')
minus = Button(label = '-')
plus.on_click(slider_plus)
minus.on_click(slider_minus)

wf_layout = layout(children = [show_peaks, cur_file_box, row(minus, seg_slider, plus), rbg, save_seg_button])
wf_layout.children.append(p_seg)
wf_layout.children.append(p_wf_II)
wf_tab = Panel(child = wf_layout, title = 'Waveforms')

##################################  Bokeh Output ##################################
eng = matlab.engine.start_matlab();
eng.addpath(r'./WFDB'); # location of ecgpuwave matlab functions
eng.addpath(r'./mcode'); 
# combine the panels and plot
layout = Tabs(tabs=[ file_tab, vs_tab, wf_tab])

curdoc().add_root(layout)
