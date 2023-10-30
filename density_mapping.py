# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 10:30:05 2022

@author: Lukas Wimmer
"""


from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import seaborn as sns

import pims
import trackpy as tp
import sklearn.datasets as data

from rdfpy import rdf

from scipy.optimize import curve_fit
import scipy.special as scsp
from scipy.ndimage import gaussian_filter, gaussian_filter1d

import scienceplots

### --- Generate Greyscale Vertical ---- ###
       
def grayscale_v(frame,sum_step):
    count = 0
    imgshape = frame.shape
    grayscale = np.zeros(int(imgshape[0]), dtype=float)
    #Grayscale by ROW
    for row in range(0,imgshape[0]):
        sumpixel = 0
        count += 1
        for column in range(0,imgshape[1]):
            sumpixel += frame[row][column]
        grayscale[int(row)] = sumpixel/int(imgshape[0]);
    #Step-wise grayscale    
    count = 0
    sumpixel = 0
    for i in range(len(grayscale)):
        sumpixel += grayscale[i]
        grayscale[i] = 0
        if count == sum_step-1:
            grayscale[i-int(sum_step/2)] = sumpixel
            count = 0
            sumpixel = 0
        count += 1     
    return grayscale

### --- Interparticle Distance Analysis --- ###

def interparticle_distance(dataframe, particle_dia, min_mass_cut, iter_step_dr):
    #
    df_located = tp.locate(dataframe,particle_dia,minmass=min_mass_cut)
    #
    x_coords = df_located['x'].to_numpy()
    y_coords = df_located['y'].to_numpy()
    #
    #PLOT
    #test_data = np.transpose(np.vstack((x_coords,y_coords)))
    #fig,ax = plt.subplots(dpi=600)
    #plt.imshow(dataframe)
    #plt.scatter(test_data.T[0], test_data.T[1], color='r', marker='o',s=30,linewidths=1, facecolors='none')
    #
    cart_coords = np.transpose(np.vstack((x_coords,y_coords)))
    #
    g_r3, radii3 = rdf(cart_coords, dr=iter_step_dr, parallel=False)
    #
    g_r3 = gaussian_filter1d(g_r3, sigma=2)
    #
    #PLOT
    with plt.style.context(['science','no-latex','grid']):
        fig,ax = plt.subplots(figsize=(5,2.5),dpi=800)
        #
        ax.plot(radii3, g_r3)
        #
        #Axes
        plt.xlabel('x[Pixel]')
        plt.ylabel('$g_r$')
    #
    peak = np.where(g_r3 == np.amax(g_r3))
    #
    #print(peak[:])
    #
    if len(peak[0]) != 0:
        result_in_px = radii3[peak[0]][0]
        result_in_mm = radii3[peak[0]][0] * 0.0118 #mm
    else:
        result_in_mm = result_in_px = 0
    #
    #print("From " + str(len(x_coords)) + " detected particle")
    #print("Average interparticle distance = "+str(result_in_mm)+" mm")
    #
    #print(result_in_px)
    #
    return result_in_px

def density_2D_v_modeling(frames, step_width, overlap, ps_in_px, min_mass_cut, iter_step_dr):
    #prepare frame slices
    interparticle_distance_tp = [] 
    grayscale = []
    for i in range(len(frames)):
        frame = frames[i]
        start_point = 1100
        end_point = 1450
        if i > 3:
            start_point = start_point +200
            end_point = end_point +200
        id_temp = []
        id_temp_nozeros = []
        while start_point + step_width <= end_point:
            x = interparticle_distance(frame[start_point:(start_point + step_width),0:2000], ps_in_px, min_mass_cut, iter_step_dr)
            id_temp = np.append(id_temp, x)
            start_point = start_point + (step_width - overlap)
        for n in range(len(id_temp)):
            if id_temp[n] != 0:
                id_temp_nozeros = np.append(id_temp_nozeros, id_temp[n])
        print('frame finished '+str(i))
        interparticle_distance_tp = np.append(interparticle_distance_tp, np.average(id_temp_nozeros))   
        grayscale = np.append(grayscale, np.average(grayscale_v(frame[start_point:,0:2000], step_width)))
    interparticle_distance_tp[:] = 3/(4*np.pi*((interparticle_distance_tp[:]*0.00118)/1.79)**3)    #Calculating particle density from wigner-seitz -> a = (x*0.0018)/1.79 #in cm and density=3/(4*np.pi*a**3)
    return interparticle_distance_tp, grayscale

def density_2D_vsf(frame, step_width, overlap, start_point, end_point, ps_in_px, min_mass_cut, iter_step_dr):
    #prepare frame slices
    density = [] 
    #grayscale = []
    start_point_var = start_point
    values_axes = []
    while start_point_var + step_width <= end_point:
        x = interparticle_distance(frame[start_point_var:(start_point_var + step_width),:], ps_in_px, min_mass_cut, iter_step_dr)
        a = (x*0.00118)/1.79     #Calculating particle density from wigner-seitz -> a = (x*0.00118)/1.79 #in cm and density=3/(4*np.pi*a**3); doi = {10.1063/1.4914468}
        #error: 1.79 /pm 0.07
        density = np.append(density, 3/(4*np.pi*(a**3)))    
        print (str(a*10000)+' mum')
        values_axes = np.append(values_axes, start_point_var+overlap/2)
        start_point_var = start_point_var + (step_width - overlap)
    #
    #grayscale = np.append(grayscale, grayscale_v(frame[start_point:end_point,0:2000], step_width-overlap))
    #PLOT
    #fig,ax = plt.subplots(dpi=600)
    #
    #ax.plot(values_axes, density, 'X')
    #
    #Axes
    #plt.xlabel('x[Pixel]')
    #plt.ylabel('n_0d[cm**-3]')
    #
    print('frame finished')
    #
    return density, values_axes

#%%

### PREPERATION ###

#Frame pre analysis. Prepare frame for density calculations. Crop Image at particle population area.
frames = pims.open('densitydata/*.bmp')

img1 = frames[0]

cut_l = 1250
cut_r = 1650

img_1 = img1[cut_l:cut_r,:]


# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')
plt.imshow(img_1)

#%%

img_1 = gaussian_filter(img_1, sigma=1)
plt.imshow(img_1)

#%%
#https://michael-fuchs-python.netlify.app/2020/06/20/hdbscan/

#PSF = 3-12 depending on particle size and exposure time
#Brigthness Threshold = cancel out faint particles, depending on camera gain and exposure time
#doi = {10.1063/1.4914468}  

df_located = tp.locate(img_1,9,70)      #(img, PSD, min_brigthness)

#
x_coords = df_located['x'].to_numpy()
y_coords = df_located['y'].to_numpy()
#
data = np.transpose(np.vstack((x_coords,y_coords)))

#PLOT
plot_kwds = {'alpha' : 0.25, 's' : 25, 'linewidths':1}
fig,ax = plt.subplots(dpi=600)
plt.imshow(img_1)
plt.scatter(data.T[0], data.T[1], color='r', **plot_kwds, facecolors='none')

#%%

### MAIN CODE ###

### 1.31 micrometer particles do have a psf of around 3 pixels [M.Y. Pustylnik et. al. (2016), "Plasmakristall-4: New complex (dusty) plasma laboratory on board the International Space Station"]
ps_in_px = 9 # 9   #needs to be odd number (13 from agglomeration 24.06.2022, 9 from density 16.02.2023)
min_mass_cut = 70 # 68  #from experience (200 from agglomeration 24.06.2022, 70 from density 16.02.2023)
iter_step_dr = 0.5  #from experience (accuracy and computationsla time taken into account)


#interparticle_distance_arr, grayscale = density_2D_v_modeling(frames[:-2], 100, 50, ps_in_px, min_mass_cut, iter_step_dr)
#print(interparticle_distance_arr)
#
start_point = 1250 #in pixel
end_point = 1650 #in pixel
#
start_point_1 = 1050 
end_point_1 = 1450
#
start_point_3 = 1100
end_point_3 = 1400
#
start_point_4 = 1200
end_point_4 = 1700

step_width = 150
overlap = 100

#gives particle densities in certain split sections adjustable in pixel. 
y0,x0 = density_2D_vsf(gaussian_filter(frames[0], sigma=1), step_width, overlap, start_point, end_point, ps_in_px, min_mass_cut, iter_step_dr)
y1,x1 = density_2D_vsf(gaussian_filter(frames[1], sigma=1),  step_width, overlap, start_point, end_point, ps_in_px, min_mass_cut, iter_step_dr)
y2,x2 = density_2D_vsf(gaussian_filter(frames[2], sigma=1),  step_width, overlap, start_point, end_point, ps_in_px, min_mass_cut, iter_step_dr)
y3,x3 = density_2D_vsf(gaussian_filter(frames[3], sigma=1),  step_width, overlap, start_point_3, end_point_3, ps_in_px, min_mass_cut, iter_step_dr)
y4,x4 = density_2D_vsf(gaussian_filter(frames[4], sigma=1),  step_width, overlap, start_point_4, end_point_4, ps_in_px, min_mass_cut, iter_step_dr)
y5,x5 = density_2D_vsf(gaussian_filter(frames[5], sigma=1),  step_width, overlap, start_point_4, end_point_4, ps_in_px, min_mass_cut, iter_step_dr)

#%%

#Var prep for plot
# y in cm^-3
y0 = np.multiply(y0,10**(-5))
y1 = np.multiply(y1,10**(-5))
y2 = np.multiply(y2,10**(-5))
y3 = np.multiply(y3,10**(-5))
y4 = np.multiply(y4,10**(-5))
y5 = np.multiply(y5,10**(-5))
# x in mm  from tube axis
x0 = np.multiply(np.subtract(x0,1050), 0.0118)
x1 = np.multiply(np.subtract(x1,1050), 0.0118)
x2 = np.multiply(np.subtract(x2,1050), 0.0118)
x3 = np.multiply(np.subtract(x3,1050), 0.0118)
x4 = np.multiply(np.subtract(x4,1050), 0.0118)
x5 = np.multiply(np.subtract(x5,1050), 0.0118)
#%%

#PLOT2

fig,ax = plt.subplots(figsize=(6,3),dpi=800)

ax.scatter(x0, y0, marker='^', color='#00429d', linewidth=.7, s=5, facecolors='none')
ax.scatter(x1, y1, marker='s', color='#00cc00', linewidth=.7, s=5, facecolors='none')
ax.scatter(x2, y2, marker='o', color='#ff8000', linewidth=.7, s=5, facecolors='none')
ax.scatter(x3, y3, marker ='x', color='#ff0000', linewidth=.7, s=5)
ax.legend(['15 Pa','20 Pa', '25 Pa', '30 Pa'], loc='upper right', prop={'size': 9})

plt.xlabel('r[mm]')
plt.ylabel('$n_{d}$ [$10^5 cm^{-3}$]')

ax.grid(color='grey', linestyle='--', linewidth=0.4, alpha=0.5)
plt.show()
