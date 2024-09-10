#include "clustream.hpp"

PYBIND11_MODULE(clustream, m) {
    py::class_<CluStream>(m, "__CluStream__")
        .def(py::init<int, int, int>(), py::arg("h") = 100, py::arg("m") = 1000, py::arg("t") = 2)
        .def_readonly("m", &CluStream::m)
        .def_readonly("time_window", &CluStream::time_window)
        .def_readonly("t", &CluStream::t)
        .def_readonly("timestamp", &CluStream::timestamp)
        .def_readonly("points_fitted", &CluStream::points_fitted)
        .def_readonly("points_forgot", &CluStream::points_forgot)
        .def_readonly("points_merged", &CluStream::points_merged)
        .def("batch_online_cluster", &CluStream::batch_online_cluster)
        .def("partial_fit", &CluStream::batch_online_cluster)
        .def("get_kernel_centers", &CluStream::get_kernel_centers)
        .def("get_partial_cluster_centers", &CluStream::get_kernel_centers)
        .def("init_kernels_offline", &CluStream::init_kernels_offline)
        ;
}
