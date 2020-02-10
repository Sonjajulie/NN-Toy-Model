#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 12:13:42 2019

@author: sonja
"""
import seaborn
import matplotlib.pyplot as plt

class PlotDataPoints():
    """ Plot true and modelled data"""
    def __init__(self,label):
        self.label = label
    
    def plot_data_true_modelled(self,time,true_data,modelled_data,share_y=False,label=None):
        """ 
        Create three subplots with each representing one point of
        forecast data and true data
        """
        nmb_pnts = len(modelled_data)
        plt.style.use('seaborn')
        # nrows, ncolumns
        axes=tuple([f"ax{i}" for i in range(1,nmb_pnts+1)])

        fig, axes = plt.subplots(nmb_pnts,1, sharey=share_y, sharex=True,figsize=(20,20))
        fig.suptitle("Performance of NN", fontsize=16)

        for i in range(nmb_pnts):
            axes[i].plot(time, modelled_data[i],'-',label='modelled data')
            axes[i].plot(time,true_data[i],'-',label='True data', alpha=0.5)
            
            # set ticks and x labels
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_xlabel("Time", fontsize=12)
            axes[i].tick_params(axis='both', which='major', labelsize=14)
        # set y label and legend
        axes[0].set_ylabel("Predictand", fontsize=14)
        axes[-1].legend()
        plt.savefig(f"{self.label}", bbox_inches='tight')


    def plot_data(self,time,data,label=None):
        """ 
        Create three subplots with each representing one point of
        forecast data and true data
        """
        if label!=None:
            self.label=label
            
        nmb_pnts = len(data)
        plt.style.use('seaborn')
        # nrows, ncolumns
        axes=tuple([f"ax{i}" for i in range(1,nmb_pnts+1)])
        
        fig, axes = plt.subplots(nmb_pnts,1, sharex=True,figsize=(20,20))
        fig.suptitle("Performance of NN", fontsize=16)

        for i in range(nmb_pnts):
            axes[i].plot(time, data[i],'-',label='data')
            # set ticks and x labels
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_xlabel("Time", fontsize=12)
            axes[i].tick_params(axis='both', which='major', labelsize=14)
        # set y label and legend
        axes[0].set_ylabel("data", fontsize=14)
        axes[-1].legend()
        plt.savefig(f"{self.label}", bbox_inches='tight')
        
if __name__=='__main__':
    time = range(1, 1001)
    true_data = []
    modelled_data = []
    true_data.append( [x**2 for x in time])
    true_data.append([x**3 for x in time])
    true_data.append([x**4 for x in time])
    modelled_data.append( [x for x in time])
    modelled_data.append([x**2 for x in time])
    modelled_data.append([x**3 for x in time])
    pdp=PlotDataPoints("Test.png")
    pdp.plot_data_true_modelled(time,true_data,modelled_data,label="Test2.png")
    pdp.plot_data(time,true_data,label="Test_solo.png")