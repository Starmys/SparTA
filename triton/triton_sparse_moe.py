# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import List

import torch
import triton
import triton.language as tl

from sparta.nn import DynamicSparseMoE
from sparta.testing import profile
import sparse_moe_cpp


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_moe_kernel(
    inputs, weights, expert_count, sparse_index, outputs,
    M, N, K,
    stride_inputs_m, stride_inputs_k,
    stride_weights_e, stride_weights_k, stride_weights_n,
    stride_count_e,
    stride_index_e, stride_index_m,
    stride_outputs_m, stride_outputs_n,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    exp_id = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    max_m = tl.load(expert_count + exp_id * stride_count_e)
    if pid_m * BLOCK_SIZE_M >= max_m:
        return
    idx_m = tl.load(sparse_index + exp_id * stride_index_e + offs_am * stride_index_m)
    a_ptrs = inputs + (idx_m[:, None] * stride_inputs_m + offs_k[None, :] * stride_inputs_k)
    b_ptrs = weights + (exp_id * stride_weights_e + offs_k[:, None] * stride_weights_k + offs_bn[None, :] * stride_weights_n)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a_mask = (offs_am[:, None] < max_m) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_inputs_k
        b_ptrs += BLOCK_SIZE_K * stride_weights_k
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = outputs + stride_outputs_m * idx_m[:, None] + stride_outputs_n * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < max_m) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_sparse_moe(
    inputs: torch.Tensor,  # [M, K]
    weights: torch.Tensor,  # [num_exps, K, N]
    exp_ids: torch.Tensor,  # [M]
):
    M, K = inputs.shape
    num_exps, K, N = weights.shape

    sparse_index = torch.zeros(num_exps, 4096, dtype=torch.int32, device=inputs.device)
    expert_count = torch.zeros(num_exps, dtype=torch.int32, device=inputs.device)
    sparse_moe_cpp.convert_index(exp_ids, sparse_index, expert_count)

    outputs = torch.empty((M, N), device=inputs.device, dtype=inputs.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), num_exps,
    )
    triton_moe_kernel[grid](
        inputs, weights, expert_count, sparse_index, outputs,
        M, N, K,
        inputs.stride(0), inputs.stride(1),
        weights.stride(0), weights.stride(1), weights.stride(2),
        expert_count.stride(0),
        sparse_index.stride(0), sparse_index.stride(1),
        outputs.stride(0), outputs.stride(1),
    )
    return outputs


def moe_reference(
    exp_modules: List[torch.nn.Linear],
    data: torch.Tensor,
    exp_ids: torch.Tensor,
    out_dims: int,
):
    n_exp = len(exp_modules)
    out = torch.zeros((data.size(0), out_dims), dtype=data.dtype, device=data.device)
    for eid in range(n_exp):
        out[exp_ids == eid] = exp_modules[eid](data[exp_ids == eid])
    return out


def profile_sparse_moe(
    batch: int = 32,
    seq_len: int = 128,
    num_exps: int = 16,
    in_dims: int = 4096,
    out_dims: int = 4096,
    dtype: torch.dtype = torch.float16,
):
    torch.manual_seed(2022)
    exp_modules = [
        torch.nn.Linear(in_dims, out_dims, bias=False, dtype=dtype, device='cuda')
        for _ in range(num_exps)
    ]
    sparta_moe = DynamicSparseMoE(exp_modules)

    data = torch.rand((batch * seq_len, in_dims), dtype=dtype, device='cuda')
    exp_ids = torch.randint(0, num_exps, (batch * seq_len, ), dtype=torch.int32, device='cuda')

    moe_weights = sparta_moe.weight.detach()

    sparta_out = sparta_moe(data, exp_ids)
    triton_out = triton_sparse_moe(data, moe_weights, exp_ids)
    target_out = moe_reference(exp_modules, data, exp_ids, out_dims)

    torch.testing.assert_close(sparta_out, target_out, atol=2e-2, rtol=2e-4)
    torch.testing.assert_close(triton_out, target_out, atol=1e-3, rtol=1e-4)

    title = f'MxNxK={batch * seq_len}x{out_dims}x{in_dims}, num_exps={num_exps}, dtype={dtype}'
    sparta_latency = profile(moe_reference, [exp_modules, data, exp_ids, out_dims], num_warmups=200, num_iters=1000)
    print(f'[{title}] dense latency: {sparta_latency} ms')
    sparta_latency = profile(sparta_moe, [data, exp_ids], num_warmups=200, num_iters=1000)
    print(f'[{title}] sparta latency: {sparta_latency} ms')
    triton_latency = profile(triton_sparse_moe, [data, moe_weights, exp_ids], num_warmups=200, num_iters=1000)
    print(f'[{title}] triton latency: {triton_latency} ms')


if __name__ == '__main__':
    profile_sparse_moe()
