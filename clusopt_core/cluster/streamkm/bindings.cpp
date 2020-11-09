#include "wrapper.hpp"

#define INIT_DOC \
"StreaKM++ data stream clustering algorithm implementation\n\
\n\
Args:\n\
	coresetsize (int): Number of coresets to use\n\
	length (int): Total length of the dataset\n\
Attributes:\n\
	coresetsize (int):\n\
	length (int):\n\
"
#define BATCH_ONLINE_CLUSTER_DOC \
"Process a chunk of datapoints all at once\n\
\n\
Args:\n\
	batch (ndarray): the datapoint chunk matrix\n"
#define GET_STREAMING_CORESET_CENTERS_DOC \
"Get current streaming coreset centers\n\
\n\
Returns:\n\
	ndarray\n"

PYBIND11_MODULE(streamkm, m) {
	py::class_<Streamkm>(m, "Streamkm",INIT_DOC)
		.def(py::init<int,int,int>(),py::arg("coresetsize"),py::arg("length"),py::arg("seed"))
		.def_readonly("coresetsize",&Streamkm::coresetsize)
		.def_readonly("length",&Streamkm::length)
		.def("batch_online_cluster",&Streamkm::batch_online_cluster,BATCH_ONLINE_CLUSTER_DOC)
		.def("partial_fit",&Streamkm::batch_online_cluster,BATCH_ONLINE_CLUSTER_DOC)
		.def("get_streaming_coreset_centers",&Streamkm::get_streaming_coreset_centers,GET_STREAMING_CORESET_CENTERS_DOC)
		.def("get_partial_cluster_centers",&Streamkm::get_streaming_coreset_centers,GET_STREAMING_CORESET_CENTERS_DOC)
	;
}