#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include <vector>
#include <cmath>
#include <limits>

typedef py::array_t<double, py::array::c_style | py::array::forcecast> ndarray_double;
typedef py::array_t<int, py::array::c_style | py::array::forcecast> ndarray_int;


typedef unsigned int uint;

#define INF std::numeric_limits<double>::infinity()

#define DOUBLE_NAN std::numeric_limits<double>::quiet_NaN()

class Accumulator {

private:
    double sum_;
    uint n;

public:
    Accumulator() :sum_(0), n(0) {}

    void add(double value) {
        sum_ += value;
        n++;
    }

    double mean() {
        if (n == 0)return DOUBLE_NAN;
        return (double)sum_ / n;
    }

    void reset() {
        sum_ = 0;
        n = 0;
    }
};

class Silhouette {

private:
    std::vector<Accumulator> cluster_table;
    Accumulator score_accum;
    uint k;

public:
    Silhouette(uint n_clusters) :
        cluster_table(n_clusters),
        score_accum(),
        k(n_clusters)
    {}

    double get_score(uint no_samples, double* dist_table, int* labels) {
        int* label_i = labels;
        double* dist_vec = dist_table;

        for (uint i = 0;i < no_samples;i++) {
            //----------------------------------------------------
            //compute ai,bi

            int* label_j = labels;
            for (uint j = 0;j < no_samples;j++) {
                if (i != j)
                    cluster_table[*label_j].add(*dist_vec);
                dist_vec++; //will iterate no_samples X no_samples
                label_j++;
            }


            double ai = cluster_table[*label_i].mean();
            double bi = INF;

            for (uint j = 0;j < k;j++) {
                double m = cluster_table[j].mean();
                cluster_table[j].reset(); // take advantage of iteration
                if (j == (uint)*label_i)continue;
                if (m < bi)
                    bi = m;
            }

            //----------------------------------------------------
            double s = (bi - ai) / fmax(ai, bi);
            score_accum.add(std::isnan(s) ? 0 : s);
            label_i++;
        }
        double ans = score_accum.mean();
        score_accum.reset();
        return ans;
    }

    double get_score_py(ndarray_double dist_table, ndarray_int labels) {
        auto dist_table_buff = dist_table.request();
        auto labels_buff = labels.request();

        if (dist_table_buff.ndim != 2)
            throw std::runtime_error("dist_table must be a matrix");
        if (dist_table_buff.shape[0] != dist_table_buff.shape[1])
            throw std::runtime_error("dist_table must be a square matrix");
        if (labels_buff.ndim != 1)
            throw std::runtime_error("labels must be a vector");
        if (labels_buff.shape[0] != dist_table_buff.shape[0])
            throw py::index_error("inconsistent number of labels");

        return get_score(
            labels_buff.shape[0],
            (double*)dist_table_buff.ptr,
            (int*)labels_buff.ptr
        );
    }
};
PYBIND11_MODULE(silhouette, m) {
    py::class_<Silhouette>(m, "Silhouette")
        .def(py::init<int>(), py::arg("n_clusters"))
        .def("get_score", &Silhouette::get_score_py)
        ;
}