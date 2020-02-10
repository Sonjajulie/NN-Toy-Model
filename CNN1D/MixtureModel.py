#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 14:57:23 2019

@author: sonja
"""

from scipy.stats import multivariate_normal
import numpy as np


class MixtureGaussianModel:
    """ Create a mixture gaussian model"""
    sigma: float

    def __init__(self, gaussian_distribution_models):
        """ add all models by given mean and sigma as well as dimensions of data
        and number of models
        """
        self.submodels = []
        self.len_models = 0
        self.res = 0
        self.submodel_samples = {}
        self.submodel_choices = []
        self.dim = len(gaussian_distribution_models[0]["mean"])
        for dist in gaussian_distribution_models:
            self.submodels.append(multivariate_normal(dist["mean"], dist["sigma"]))
            self.len_models += 1

    def rvs(self, size):
        """ get samples of mixture gaussian model"""
        self.submodel_samples = np.zeros((size, self.dim))
        np.random.seed(0)
        self.submodel_choices = np.random.randint(self.len_models, size=size)
        for idx, sample in enumerate(self.submodel_choices):  # random_state=12345
            self.submodel_samples[idx] = (self.submodels[sample].rvs())
        return self.submodel_samples

    def rvs_2d(self, size_time, size_spatial):
        """ get samples of mixture gaussian model"""

        # save not only precursor data, but also mean and std which were used to calc precursors
        self.submodel_samples = {'X': np.zeros((size_time, size_spatial)),
                                 'X_feature': np.zeros((size_time,size_spatial)), # np.zeros((self.dim//2,size_time * size_spatial//self.dim*2))
                                 'mean': np.zeros((size_time, self.dim)),
                                 'std': np.zeros((size_time, self.dim))}  # np.zeros((size_time, size_spatial))
        np.random.seed(0)
        self.submodel_choices = np.random.randint(self.len_models, size=size_time)
        len_pts = (size_spatial) // self.dim

        for idx_time, ch in enumerate(self.submodel_choices):  # random_state=12345
            sample = (self.submodels[ch].rvs())
            pt: float
            self.submodel_samples['mean'][idx_time] = sample
            self.submodel_samples['std'][idx_time] = np.random.normal(0.2, 0.05)

            idx1 = 0
            idx2 = 0
            for idxPt, pt in enumerate(sample):
                # for divide by number of points I want to have in total
                # times different composites since otherwise it will be
                # composites times too many points
                # for idx_spatial, x in enumerate(np.arange(start, end, self.res)):  # random_state=12345

                self.submodel_samples['X'][idx_time, idxPt * 250:idxPt * 250+250] = np.random.normal(pt, self.submodel_samples['std'][idx_time,idxPt] , len_pts)

        return self.submodel_samples


# unit tests: check sub_model choices
#             check numbers of submodel-generator

if __name__ == '__main__':
    # create gaussian characterics from which we want to sample
    gaussian_distributions = [
        {"mean": [-1, 1, 1, -1], "sigma": [[0.1, 0., 0., 0.], [0., 0.1, 0., 0.], [0., 0., 0.1, 0.], [0., 0., 0., 0.1]]},
        {"mean": [-1, 0, 1, 1], "sigma": [[0.1, 0., 0., 0.], [0., 0.1, 0., 0.], [0., 0., 0.1, 0.], [0., 0., 0., 0.1]]},
    ]

    # create instance to get samples
    mgm = MixtureGaussianModel(gaussian_distributions)
    print(mgm.rvs(5))
    print(mgm.dim)
