import os,sys;sys.path.append(os.path.abspath("/home/kmm11/pycrysfml/hklgen/"))
import numpy as N
import pylab
import matplotlib.pyplot as plt
from .sgolay import savitzky_golay as savitzky_golay
import sys
import scipy.interpolate as interpolate
import csv
import random
pi=N.pi

def findmin(x,y,npeaks):
    """This is a program that finds the positions and FWHMs in a set of
    data specified by x and y.  The user supplies the number of peaks and
    the program returns an array p, where the first entries are the positions of
    the peaks and the next set are the FWHMs of the corresponding peaks
    The program is adapted from a routine written by Rob Dimeo at NIST and
    relies on using a Savit-Golay filtering technique to obtain the derivative
    without losing narrow peaks.  The parameter F is the frame size for the smoothing
    and is set to 11 pts.  The order of the polynomial for making interpolations to better
    approximate the derivative is 4.  I have improved on Dimeo's program by also calculating second
    derivative information to better handle close peaks.  If peaks are too close together, there are
    still problems because the derivative may not turn over.  I have also added a refinement of going
    down both the left and right sides of the peak to determine the FWHMs because of the issue of peaks that
    are close together."""

    y=N.array(y)
    x=N.array(x)
    step=abs(x[0]-x[1]) #assume that x is monotonic and uniform step sizes
    yd=savitzky_golay(y,deriv=1)/step
    yd2=savitzky_golay(y,deriv=2)/step**2
    n_crossings=0;
    ny = len(yd);
    value_sign = 2*(yd > 0) - 1;
    indices = 0;

    # Determine the number of zero crossings
    diff_sign=N.hstack(([0],N.diff(value_sign)))

    # wh_cross = find(((diff_sign==2) | (diff_sign==-2)) & yd2<0);
    wh_cross_table=N.abs(diff_sign)==2 
    yd_table=yd2>0

    index_list=N.array(range(len(wh_cross_table)))

    wh_cross=index_list[N.array(wh_cross_table)*N.array(yd_table)]
    wh_cross=wh_cross.astype(int)

    n_crossings=len(wh_cross);

    indices = 0.5*(2*wh_cross-1);
    indices=wh_cross
    no_width = 0;
  
    print("indicies outside of else: ", indices)

    if n_crossings>0:
        ysupport=range(len(y))
        yinterpolater=interpolate.interp1d(ysupport,y,fill_value=0.0,kind='linear',copy=True,bounds_error=False)
        ymax=yinterpolater(indices)
        ymin = N.min(ymax)
        this_max=N.min(ymax)
        max_index=N.nonzero(ymax==this_max)
        best_index = indices[max_index]
        print('>>>>>>>>>>>>>>>best', best_index)
    else:
        best_index = 0

    return best_index

def fp_gaussian(x,area,center,fwhm):
    sig = fwhm/2.354;
    y = (area/N.sqrt(2.0*pi*sig**2))*N.exp(-0.5*((x-center)/sig)**2);
    return y

def matlab_gaussian(x,p):
    area,center,fwhm=p
    sig = fwhm/2.354;
    #y= N.abs(I)*N.exp(-0.5*((x-center)/w)**2)
    y = (area/N.sqrt(2.0*pi*sig**2))*N.exp(-0.5*((x-center)/sig)**2);
    return y

if __name__=="__main__":
    
    with open('/home/jpr6/logs/bumps_tests/75/dat_random_set14.csv') as f:
        reader = csv.reader( f, delimiter = ',')
        table = [row for row in reader]
        table = N.float_(table)
        x = table[0]
        y = table[1]
    
    
    xpeaks = findmin(x,y,10)
    if 1:
        fig, ax = plt.subplots()
        plt.plot(x,y,'s')
        ax.set_xticks(x[xpeaks], minor=False)
        ax.xaxis.grid (True, which = 'major')
        plt.show()
   