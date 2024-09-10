#include "wrapper.hpp"

PYBIND11_MODULE(streamkm, m) {
	py::class_<Streamkm>(m, "__Streamkm__")
		.def(py::init<int,int,int>(),py::arg("coresetsize"),py::arg("length"),py::arg("seed"))
		.def_readonly("coresetsize",&Streamkm::coresetsize)
		.def_readonly("length",&Streamkm::length)
		.def("batch_online_cluster",&Streamkm::batch_online_cluster)
		.def("partial_fit",&Streamkm::batch_online_cluster)
		.def("get_streaming_coreset_centers",&Streamkm::get_streaming_coreset_centers)
		.def("get_partial_cluster_centers",&Streamkm::get_streaming_coreset_centers)
	;
}
