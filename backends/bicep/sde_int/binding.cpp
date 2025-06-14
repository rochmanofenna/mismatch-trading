#include <torch/extension.h>
#include <curand_kernel.h>

void sde_curand_launch(
    torch::Tensor paths,
    int n_steps,
    int stride,
    float T,
    float feedback_value,
    float decay_rate,
    float high_threshold,
    float low_threshold,
    float total_steps,
    float base_variance
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sde_curand", &sde_curand_launch, "SDE with CURAND");
}

void sde_curand_launch(...) {
    auto paths_data = paths.data_ptr<float>();
    int64_t n_paths = paths.size(0);
    int threads = 256;
    int blocks  = (n_paths + threads - 1) / threads;
    sde_curand_kernel<<<blocks,threads>>>(
      paths_data, n_steps, stride, T,
      feedback_value, decay_rate,
      high_threshold, low_threshold,
      total_steps, base_variance
    );
    cudaCheckError(); 
}
