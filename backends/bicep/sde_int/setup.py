from torch.utils.cpp_extension import load

sde_ext = load(
    name="sde_ext",
    sources=["curand_kernel.cu", "binding.cpp"],
    extra_cuda_cflags=["-O3"],
)

paths = torch.zeros((n_paths, stride), device="cuda", dtype=torch.float32)
sde_ext.sde_curand(
    paths, n_steps, stride,
    1.0, feedback_value, decay_rate,
    high_threshold, low_threshold,
    float(n_steps), base_variance
)
