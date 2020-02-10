#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 12:16:56 2019

@author: sonja
"""
# needed for multivariate distribution

from scipy.stats import multivariate_normal


class Precursors:
    """ Class to create Precursor data """
 
    def __init__(self,composites,arrays):
        """ 
        Initialize arrays, composites for data creation
        arrays - Arrays which lead with composites to clusters
        composites - composites of precursors
        """
        self.composites = composites
        self.arrays = arrays


