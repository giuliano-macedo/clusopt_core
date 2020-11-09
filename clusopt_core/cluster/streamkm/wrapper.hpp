#pragma once

#include "original/StreamingCoreset.h"
#include "original/Point.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// typedef py::array_t<double> ndarray;

typedef py::array_t<double, py::array::c_style | py::array::forcecast> ndarray; //to force c style

class Streamkm {
public:
    unsigned int coresetsize;
    unsigned int length;

    Streamkm(unsigned int coresetsize, unsigned int length, unsigned int seed);
    ~Streamkm();
    void batch_online_cluster(ndarray batch);
    ndarray get_streaming_coreset_centers();
private:
    unsigned int dim;
    struct Bucketmanager manager;
    unsigned int timestamp;
    struct point last_point;
};