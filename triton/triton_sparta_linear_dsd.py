# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, Tuple, Type, Optional
import warnings

import torch
import triton
import triton.language as tl

from sparta.kernels import SparseMatMulKernel, SparTASparseMatMulKernel, OpenAISparseMatMulKernel
from sparta.tesa import BCSIndexes
from sparta.testing import block_mask, profile


def prepare_data(
    batch: Optional[int] = 4,
    M: int = 128,
    K: int = 256,
    N: int = 192,
    granularity: Tuple[int, int] = (8, 8),
    sparsity: float = 0.9,
    mode: str = 'dds',
    trans_A: bool = False,
    trans_B: bool = False,
    biased: bool = False,
    requires_grad: bool = False,
    mask: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.float32,
    random_seed: int = 2022,
):
    inputs = ['A', 'B']
    outputs = ['C']
    shapes = {
        'A': (K, M) if trans_A else (M, K),
        'B': (N, K) if trans_B else (K, N),
        'C': (M, N),
    }
    if biased:
        inputs.append('bias')
        shapes['bias'] = (N, )

    torch.manual_seed(random_seed)
    data: Dict[str, torch.Tensor] = {}
    for x in inputs:
        shape = shapes[x] if batch is None else (batch, *shapes[x])
        data[f'input_{x}'] = torch.rand(size=shape, device='cuda', dtype=dtype)
    if requires_grad:
        for y in outputs:
            shape = shapes[y] if batch is None else (batch, *shapes[y])
            data[f'input_grad_{y}'] = torch.rand(size=shape, device='cuda', dtype=dtype)

    sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
    if mask is None:
        mask = block_mask(
            shape=shapes[sparse_port],
            granularity=granularity,
            sparsity=sparsity,
            device='cuda',
        )
    add_mask(data, mask, sparse_port, 'input')

    calc_target_data(data, requires_grad, trans_A, trans_B)
    add_mask(data, mask, sparse_port, 'target')

    return data, mask


def calc_target_data(
    data: Dict[str, torch.Tensor],
    requires_grad: bool,
    trans_A: bool,
    trans_B: bool,
):
    if requires_grad:
        for k, v in data.items():
            if k.startswith('input'):
                v.requires_grad = True

    if len(data['input_A'].shape) == 3:
        input_A = data['input_A'].swapaxes(1, 2) if trans_A else data['input_A']
        input_B = data['input_B'].swapaxes(1, 2) if trans_B else data['input_B']
        data['target_C'] = torch.bmm(input_A, input_B)
        if 'input_bias' in data:
            data['target_C'] += data['input_bias'].unsqueeze(1)
    else:
        input_A = data['input_A'].T if trans_A else data['input_A']
        input_B = data['input_B'].T if trans_B else data['input_B']
        data['target_C'] = torch.mm(input_A, input_B)
        if 'input_bias' in data:
            data['target_C'] += data['input_bias']

    if requires_grad:
        data['target_C'].backward(data['input_grad_C'])
        data['target_grad_A'] = data['input_A'].grad
        data['input_A'].grad = None
        data['target_grad_B'] = data['input_B'].grad
        data['input_B'].grad = None
        if 'input_bias' in data:
            data['target_grad_bias'] = data['input_bias'].grad
            data['input_bias'].grad = None


def add_mask(
    data: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    sparse_port: str,
    stage: str,
):
    for name, val in data.items():
        if name.startswith(stage) and name.endswith(sparse_port):
            val *= mask


def get_params(impl: str):
    if impl == 'sparta':
        return {
            '_impl': 'sparta',
            'BLOCK_SIZE_M_VALUE': 32,
            'BLOCK_SIZE_K_VALUE': 32,
            'BLOCK_SIZE_N_VALUE': 32,
        }
    else:
        return {'_impl': impl}


def compress_data(
    indexes: BCSIndexes,
    sparse_port: str,
    data: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    requires_grad: bool,
):
    for name in data:
        if name.endswith(sparse_port):
            shape = [indexes.block_nnz, indexes.block_H, indexes.block_W]
            dtype = data[name].dtype
            data[name] = indexes.convert(data[name].detach().to(torch.float32)).to(dtype).reshape(shape)
    mask = indexes.convert(mask.to(torch.float32)).to(torch.uint8)
    if sparse_port in ['A', 'B'] and requires_grad:
        data[f'input_{sparse_port}'].requires_grad = True
    return data, mask


def check_results(data: Dict[str, torch.Tensor]):
    for name, val in data.items():
        if name.startswith('target_'):
            out = data[name.replace('target', 'output')]
            # torch.testing.assert_close(out, val, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(out, val, atol=2e-2, rtol=1e-3)


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_dense_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
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
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if DTYPE == "torch.float16":
        c = accumulator.to(tl.float16)
    else:
        c = accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_dense_matmul(a, b):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    triton_dense_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        DTYPE=str(a.dtype)
    )
    return c


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_sparse_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    b_col, b_row,
    M, N, K,
    stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_b_col_m,
    stride_b_row_b,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    block_idx_start = tl.load(b_col + pid_n * stride_b_col_m)
    block_idx_end = tl.load(b_col + (pid_n + 1) * stride_b_col_m)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for block_idx in range(block_idx_start, block_idx_end):
        k = tl.load(b_row + block_idx * stride_b_row_b) & 0xffff
        a = tl.load(a_ptrs + k * BLOCK_SIZE_K * stride_ak)
        b = tl.load(b_ptrs + block_idx * stride_bb)
        accumulator += tl.dot(a, b)

    if DTYPE == "torch.float16":
        c = accumulator.to(tl.float16)
    else:
        c = accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_sparse_matmul(a, b, b_col_ptr, b_row_idx, M, N, K):
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    triton_sparse_matmul_kernel[grid](
        a, b, c,
        b_col_ptr, b_row_idx,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1),
        b_col_ptr.stride(0),
        b_row_idx.stride(0),
        DTYPE=str(a.dtype)
    )
    return c


def profile_sparse_matmul(
    impl: str = 'sparta',
    mode: str = 'dsd',
    biased: bool = False,
    compressed: bool = True,
    trans_A: bool = False,
    trans_B: bool = False,
    batch: Optional[int] = None,
    M: int = 4096,
    K: int = 4096,
    N: int = 4096,
    granularity: Tuple[int, int] = (8, 8),
    # sparsity: float = 0.995,
    sparsity: float = 0.0,
    dtype: torch.dtype = torch.float32,
):
    data, mask = prepare_data(
        batch, M, K, N,
        granularity, sparsity,
        mode, trans_A, trans_B, biased,
        False, None, dtype
    )

    sparta_kernel_class: Type[SparseMatMulKernel] = {
        'sparta': SparTASparseMatMulKernel,
        'openai': OpenAISparseMatMulKernel,
    }[impl]
    batched = batch is not None
    sparta_kernel = sparta_kernel_class(
        mode=mode,
        biased=biased,
        transpose_A=trans_A,
        transpose_B=trans_B,
        compressed=compressed,
        batched=batched,
    )
    sparta_kernel.attr.set_mask(mask)
    batch = 1 if batch is None else batch
    sparta_kernel.compile(get_params(impl), (batch, M, K, N))

    sparse_port = {'sdd': 'A', 'dsd': 'B', 'dds': 'C'}[mode]
    inputs = ['A', 'B', 'bias'] if biased else ['A', 'B']
    input_data = [data[f'input_{x}'] for x in inputs]

    indexes = sparta_kernel.attr.indexes
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        block_mask = indexes.get_block_mask()
    block_sparsity = 1 - block_mask.sum() / block_mask.numel()
    title = f'MxNxK={M}x{N}x{K}, block_sparsity={block_sparsity:.3f}, dtype={dtype}'
    # dense_latency = profile(torch.matmul, input_data, num_warmups=20, num_iters=100)
    dense_latency = profile(triton_dense_matmul, input_data, num_warmups=20, num_iters=100)
    print(f'[{title}] dense latency: {dense_latency:.3f} ms')

    if compressed:
        data, mask = compress_data(sparta_kernel.attr.indexes, sparse_port, data, mask, False)

    if dtype is torch.float32:
        input_data = [data[f'input_{x}'] for x in inputs]
        data['output_C'] = sparta_kernel(*input_data)
        add_mask(data, mask, sparse_port, 'output')
        check_results(data)
        sparta_latency = profile(sparta_kernel, input_data, num_warmups=20, num_iters=100)
        print(f'[{title}] sparta latency: {sparta_latency:.3f} ms')

    # input_data = [data[f'input_A'].half(), data[f'input_B'].half()]
    input_data = [data[f'input_{x}'] for x in inputs] + [indexes.col_ptr, indexes.BCSC_idx, M, N, K]
    data['output_C'] = triton_sparse_matmul(*input_data)
    add_mask(data, mask, sparse_port, 'output')
    check_results(data)
    triton_latency = profile(triton_sparse_matmul, input_data, num_warmups=20, num_iters=100)
    print(f'[{title}] triton latency: {triton_latency:.3f} ms')


if __name__ == '__main__':
    profile_sparse_matmul(dtype=torch.float32)
    profile_sparse_matmul(dtype=torch.float16)
