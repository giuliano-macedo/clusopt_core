#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include <limits>
#include <cstdio>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


#include "kernel.hpp"

namespace py = pybind11;

// typedef py::array_t<double> ndarray;
typedef py::array_t<double, py::array::c_style | py::array::forcecast> ndarray; //to force c style

const double double_max = std::numeric_limits<double>::max();

double c_distance(double* a, double* b, unsigned int dim);
double distance(Point& a, Point& b);

class CluStream {
public:
    std::vector<Kernel> kernels;
    int time_window;
    int m;
    int t;
    long timestamp;
    long unsigned int points_fitted;
    long unsigned int points_forgot;
    long unsigned int points_merged;

    CluStream(int h, int m, int t);
    void init_kernels_offline(ndarray cluster_centers, ndarray initpoints);
    void online_cluster(double* datapoint);

    void batch_online_cluster(ndarray batch);
    ndarray get_kernel_centers();
private:
    unsigned int dim;
};