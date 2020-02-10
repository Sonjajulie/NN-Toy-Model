#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 16:11:43 2019

@author: sonja
"""
import numpy as np
import unittest


class Predictand:
    """ Class to create Predictand data """

    def __init__(self, clusters, array_in):
        """ 
        Initialize arrays, composites for data creation
        arrays - Arrays which lead with composites to clusters
        composites - composites of precursors
        """
        self.data = []
        self.array = array_in
        self.clusters = []
        self.vv_matrix = []
        self.res = 0
        self.prcp_samples = {}
        self.dim = len(clusters[0]["cluster"])
        for cl in clusters:
            self.clusters.append(cl["cluster"])
        self.b_tensor = np.ones((self.array.shape[0], self.array.shape[1], self.array.shape[1]))

    def get_data_from_precursors(self, data_in):
        """ get predicand data from precursors"""
        self.data = np.zeros((data_in.shape[0], self.array.shape[0]))
        for idx, v in enumerate(data_in):
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html
            # tensor product of v v => matrix vv
            self.vv_matrix = np.tensordot(v, v, axes=0)
            # https://stackoverflow.com/questions/41870228/understanding-tensordot
            # in axes you define the dimensions which should be summed over
            # b_tensor sum over dims 1,2; vv_matrix sum over dims 0,1 
            # => dim 0 of b_tensor remains
            self.data[idx] = (self.array.dot(v)) + np.tensordot(self.b_tensor, self.vv_matrix, axes=((1, 2), (0, 1)))
        return self.data

    def get_data_from_precursors_2d(self, data_mean, data_std, len_time, size_spatial):
        """ get preditcand data from precursors"""
        self.prcp_samples['X'] = np.zeros((len_time, size_spatial))

        for idx_time, ch in enumerate(range(len_time)):
            self.vv_matrix = np.tensordot(data_mean[idx_time], data_mean[idx_time], axes=0)
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html
            # tensor product of v v => matrix vv
            # https://stackoverflow.com/questions/41870228/understanding-tensordot
            # in axes you define the dimensions which should be summed over
            # b_tensor sum over dims 1,2; vv_matrix sum over dims 0,1
            # => dim 0 of b_tensor remains
            sample = (self.array.dot(data_mean[idx_time])) + np.tensordot(self.b_tensor, self.vv_matrix,axes=((1, 2), (0, 1)))
            self.prcp_samples['std'] = np.mean(data_std[idx_time])
            self.size_spatial_pt = (size_spatial // self.dim)
            for idxPt, pt in enumerate(sample):
                # for divide by number of points I want to have in total
                # times different composites since otherwise it will be
                # composites times too many points
                # for idx_spatial, x in enumerate(np.arange(start, end, self.res)):  # random_state=12345
                self.prcp_samples['X'][idx_time,idxPt*(self.size_spatial_pt):
                                                idxPt*(self.size_spatial_pt) + (self.size_spatial_pt)] = \
                    np.random.normal(pt, self.prcp_samples['std'], (self.size_spatial_pt))
                        # 1 / (
                        #     self.prcp_samples['std'] * np.sqrt(2 * np.pi)) * \
                        #        np.exp(- (x - pt) ** 2 / (2*self.prcp_samples['std'] ** 2))

        return self.prcp_samples


    def get_data_point_from_precursors(self, data_point):
        """ return a single predicand point"""
        self.vv_matrix = np.tensordot(data_point, data_point, axes=0)
        return (self.array.dot(data_point)) + np.tensordot(self.b_tensor, self.vv_matrix, axes=((1, 2), (0, 1)))


class TestListElements(unittest.TestCase):
    def setUp(self):
        self.expected = [2.772124, 2.489624, 0.408124]
        self.result = list(map(lambda x: round(x, 6),
                               prcp.get_data_point_from_precursors([-1.009, 0.193, 0.940, 1.058])))

    def test_count_eq(self):
        """Check number of result list"""
        self.assertCountEqual(self.result, self.expected)

    def test_list_eq(self):
        """Check whether elements are the same"""
        self.assertListEqual(self.result, self.expected)


if __name__ == '__main__':
    prcp_clusters = [{"cluster": [-1, 1, 1, -1]}, {"cluster": [-1, 1, 1, -1]}]

    # array which lead with composites to clusters pf PRCP
    array = np.array([[1, 2, 1, 1], [-0.5, 0, -0.5, 1.], [-1, 0, -1, -1]], np.float)
    # arr_composite1 = [[1,1,1,0],[-0,0,-0.5,0.5],[-1,1,-1,0]]
    # arr_composite2 = [[0,1,0,1],[-0.5,0,0,0.5],[0,-1,0,-1]]
    prcp = Predictand(prcp_clusters, array)

    # example data snow ice:
    sce_sic = np.array([[-1.009, 0.193, 0.940, 1.058], [-0.994, 1.073, 0.909, -1.039], [-0.979, 1.062, 1.034, -0.886],
                        [-0.979, 1.062, 1.034, -0.886], [-0.979, 1.062, 1.034, -0.886]], dtype='float')

    print(prcp.get_data_from_precursors(sce_sic))
    unittest.main()
