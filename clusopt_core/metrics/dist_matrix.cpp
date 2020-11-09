#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <vector>
#include <cmath>
#include <limits>
#include <boost/thread/thread.hpp>

typedef py::array_t<double, py::array::c_style | py::array::forcecast> ndarray;

typedef unsigned int uint;

double c_distance(double* a, double* b, unsigned int dim) {
    double ans = 0;
    for (unsigned int i = 0;i < dim;i++) {
        double diff = *b - *a;
        ans += diff * diff;

        a++;
        b++;
    }
    return sqrt(ans);
}

class DistanceMatrix {

private:
    double* table_ptr;
    uint max_size;
    uint dim;
    uint no_threads;

    uint no_samples;
    double* dataset;
public:
    ndarray table;

    DistanceMatrix(uint msize, uint nthreads) :max_size(msize) {
        if (nthreads == 0) {
            no_threads = boost::thread::hardware_concurrency();
            no_threads = (no_threads == 0) ? 1 : no_threads;
        }

        table = ndarray(max_size * max_size);
        table.resize({ max_size,max_size });
        table_ptr = (double*)table.request().ptr;

        for (uint i = 0;i < max_size;i++)table_ptr[(i * max_size) + i] = 0; // main diagonal =0

    }

    void handler(uint m) {
        uint counter = 0;
        for (uint i = 0;i < no_samples;i++) {
            for (uint j = i + 1;j < no_samples;j++) {
                if (counter % no_threads == m) {
                    double dist = c_distance(
                        dataset + (i * dim),
                        dataset + (j * dim),
                        dim
                    );
                    table_ptr[(i * no_samples) + j] = dist;
                    table_ptr[(j * no_samples) + i] = dist;
                }
                counter++;
            }
        }
    }

    void compute(double* dset, uint nsamples) {
        dataset = dset;
        no_samples = nsamples;

        boost::thread_group threads;
        for (uint m = 0;m < no_threads;m++) {
            threads.add_thread(
                new boost::thread(&DistanceMatrix::handler, this, m)
            );
        }
        threads.join_all();

    }

    ndarray compute_py(ndarray dataset) {
        auto buff = dataset.request();
        if (buff.ndim != 2)
            throw std::runtime_error("dataset must be a matrix");
        dim = buff.shape[1];
        uint no_samples = buff.shape[0];
        if (no_samples > max_size)
            throw std::runtime_error("dataset larger than max_size");
        compute((double*)buff.ptr, no_samples);
        return table;
    }

};
PYBIND11_MODULE(dist_matrix, m) {
    py::class_<DistanceMatrix>(m, "DistanceMatrix")
        .def(py::init<int, int>(), py::arg("max_size"), py::arg("no_threads") = 0)
        .def("compute", &DistanceMatrix::compute_py)
        .def_readonly("table", &DistanceMatrix::table)
        ;
}