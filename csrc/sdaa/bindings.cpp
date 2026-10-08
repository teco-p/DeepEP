#include "buffer.hpp"
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>

// Main ELF pybind11_exec_teco_ep_cpp 0x197b8. Reuse pybind11's module and
// Tensor caster implementations rather than reproducing their generated code.
void bind_sdaa_routing(pybind11::module_&);
PYBIND11_MODULE(_sdaa, m) {
    m.doc() = "TecoEP: an efficient expert-parallel communacion library for Teco device";
    using teco_ep::Buffer;
    bind_sdaa_routing(m);
    pybind11::class_<Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int, int, std::size_t, bool>())
        .def("get_epoch_status", &Buffer::get_epoch_status)
        .def("get_num_nodes", &Buffer::get_num_nodes)
        .def("is_available", &Buffer::is_available)
        .def("get_dispatch_props", &Buffer::get_dispatch_props)
        .def("get_combine_props", &Buffer::get_combine_props)
        .def("get_dispatch_ipc_handle", &Buffer::get_dispatch_ipc_handle)
        .def("get_combine_ipc_handle", &Buffer::get_combine_ipc_handle)
        .def("get_dispatch_layout", &Buffer::get_dispatch_layout)
        .def("gather_dispatch_data", &Buffer::gather_dispatch_data)
        .def("get_combine_layout", &Buffer::get_combine_layout)
        .def("gather_combine_data", &Buffer::gather_combine_data)
        .def("sync", &Buffer::sync)
        .def("dispatch_data", &Buffer::dispatch_data)
        .def("combine_data", &Buffer::combine_data)
        .def("cross_buffer_put", &Buffer::cross_buffer_put)
        .def("cross_buffer_get", &Buffer::cross_buffer_get)
        .def("debug_dump_combine_headers", &Buffer::debug_dump_combine_headers);
}
