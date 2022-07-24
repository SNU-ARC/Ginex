#include <stdlib.h>
#include <aio.h>
#include <omp.h>
#include <unistd.h>
#include <fcntl.h>
#include <torch/extension.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/script.h>
#include <errno.h>
#include <cstring>
#include <inttypes.h>
#include <ATen/ATen.h>

void cache_update(torch::Tensor cache, torch::Tensor address_table, torch::Tensor batch_inputs, torch::Tensor in_indices, torch::Tensor in_positions, torch::Tensor out_indices, int64_t num_features){

        auto cache_data = cache.data_ptr<float>();
        auto address_table_data = address_table.data_ptr<int32_t>();
        auto batch_inputs_data = batch_inputs.data_ptr<float>();
        auto in_indices_data = in_indices.data_ptr<int64_t>();
        auto in_positions_data = in_positions.data_ptr<int32_t>();
        auto out_indices_data = out_indices.data_ptr<int64_t>();

        int64_t num_idx = in_indices.numel();
        int64_t feature_size = num_features*sizeof(float);

        #pragma omp parallel for num_threads(torch::get_num_threads())
        for (int64_t n = 0; n < num_idx; n++) {
                int32_t cache_out_idx = address_table_data[out_indices_data[n]];
                memcpy(cache_data+num_features*cache_out_idx, batch_inputs_data+num_features*in_positions_data[n], feature_size);
                address_table_data[in_indices_data[n]] = cache_out_idx;
                address_table_data[out_indices_data[n]] = -1;
        }

        return;
}

PYBIND11_MODULE(update, m) {
    m.def("cache_update", &cache_update, "evict & insert cache entries with the given indices");
}


