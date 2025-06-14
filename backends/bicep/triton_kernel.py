import triton
import triton.language as tl
import torch

@triton.jit
def fused_sde_kernel(path_ptr, n_steps, stride, 
                     T, directional_bias, **meta):
    pid = tl.program_id(0)
    # ptr to the start of this path
    path = path_ptr + pid * stride
    dt = T / n_steps
    acc = tl.load(path)  # initial value
    for i in range(n_steps):
        # in-kernel RNG (Philox)
        rnd = tl.random(seed=pid, i=i)
        inc = tl.log(1 + rnd) * tl.sqrt(dt)  # replace with your control logic
        acc += inc
        tl.store(path + i + 1, acc)

# Allocate a batch of 1024 paths, each of length 1000
n_paths, n_steps = 1024, 1000
paths = torch.zeros((n_paths, n_steps+1), device='cuda', dtype=torch.float32)

# Launch with one program per path
fused_sde_kernel[(n_paths,)](paths, n_steps, n_steps+1, 1.0, 0.0)
