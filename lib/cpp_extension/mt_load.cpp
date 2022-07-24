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
#define ALIGNMENT 4096

torch::Tensor load_float32(std::string file, int64_t size){

    // open file
    int fd = open(file.c_str(), O_RDONLY | O_DIRECT);

    int64_t result_buffer_size = size*sizeof(float)+ALIGNMENT;
    int64_t num_blocks = result_buffer_size / ALIGNMENT;
    float* result_buffer = (float*)aligned_alloc(ALIGNMENT, result_buffer_size);


    #pragma omp parallel for num_threads(atoi(getenv("GINEX_NUM_THREADS")))
    for (int64_t n = 0; n < num_blocks; n++) {
        int64_t offset = n*ALIGNMENT;
            
        if (pread(fd, result_buffer+(ALIGNMENT/sizeof(float))*n, ALIGNMENT, offset) == -1){
            fprintf(stderr, "load.cpp::1::ERROR: %s\n", strerror(errno));
        }
    }

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    auto result = torch::from_blob(result_buffer, {size}, options);

    close(fd);

    return result;

}


torch::Tensor load_int64(std::string file, int64_t size){

    // open file
    int fd = open(file.c_str(), O_RDONLY | O_DIRECT);

    int64_t result_buffer_size = size*sizeof(int64_t)+ALIGNMENT;
    int64_t num_blocks = result_buffer_size / ALIGNMENT;
    int64_t* result_buffer = (int64_t*)aligned_alloc(ALIGNMENT, result_buffer_size);

    #pragma omp parallel for num_threads(atoi(getenv("GINEX_NUM_THREADS")))
    for (int64_t n = 0; n < num_blocks; n++) {
        int64_t offset = n*ALIGNMENT;
            
        if (pread(fd, result_buffer+(ALIGNMENT/sizeof(int64_t))*n, ALIGNMENT, offset) == -1){
            fprintf(stderr, "load.cpp::2::ERROR: %s\n", strerror(errno));
        }
    }

    auto options = torch::TensorOptions()
        .dtype(torch::kInt64)
        .layout(torch::kStrided)
        .device(torch::kCPU)
        .requires_grad(false);
    auto result = torch::from_blob(result_buffer, {size}, options);

    close(fd);

    return result;

}

PYBIND11_MODULE(mt_load, m) {
    m.def("load_float32", &load_float32, "multi-threaded load (float32)");
	m.def("load_int64", &load_int64, "multi-threaded load (int64)");
}

