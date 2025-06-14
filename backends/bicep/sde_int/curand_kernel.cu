#include <curand_kernel.h>
#include <cuda.h>

extern "C"
__global__ void sde_curand_kernel(
    float* __restrict__ paths,
    int    n_steps,
    int    stride,
    float  T,
    float  feedback_value,
    float  decay_rate,
    float  high_threshold,
    float  low_threshold,
    float  total_steps,
    float  base_variance
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    float dt = T / n_steps;
    // initialize RNG per-thread
    curandStatePhilox4_32_10_t state;
    curand_init( /* seed */ 1234ULL, /* subsequence */ pid, /* offset */ 0, &state);
    // load initial
    float acc = paths[pid * stride];
    for (int i = 0; i < n_steps; ++i) {
        float rnd = curand_normal(&state);
        float norm    = 1.0f / total_steps;
        float norm    = 1.0f / total_steps;
        float factor1 = norm < low_threshold ? 1.5f
                                               : (norm > high_threshold ? 0.5f : 1.0f);
        float t       = i * dt;
        float vf      = base_variance * factor1 * expf(-decay_rate * t);
        float tmp    = 0.5f + feedback_value * 0.5f;
        float scale2 = fminf(1.0f, fmaxf(0.2f, tmp));
        float inc = rnd * sqrtf(dt) * scale2 * vf;
    }
}
