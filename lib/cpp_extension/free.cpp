#include <stdlib.h>
#include <torch/extension.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/script.h>
#include <ATen/ATen.h>

void tensor_free(torch::Tensor t){
    auto t_data = t.data_ptr();

    free(t_data);

    return;
}

PYBIND11_MODULE(free, m) {
    m.def("tensor_free", &tensor_free);
}
