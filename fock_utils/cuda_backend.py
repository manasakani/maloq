import torch
import torch.utils.cpp_extension
import math

# --- CUDA C++ SOURCE CODE ---
cuda_src = r'''
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void get_H_cuda_kernel(
    const scalar_t* __restrict__ net_out,
    const scalar_t* __restrict__ wms,
    scalar_t* __restrict__ H_out,
    const int* __restrict__ in_slices,
    const int* __restrict__ H_slices,
    const int* __restrict__ wm_slices,
    int num_edges,
    int total_in_dim,
    int total_h_dim) 
{
    int pid_block = blockIdx.y;
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge_idx >= num_edges) return;

    int in_start = in_slices[pid_block];
    int in_dim = in_slices[pid_block + 1] - in_start;
    int h_start = H_slices[pid_block];
    int h_dim = H_slices[pid_block + 1] - h_start;
    int wm_start = wm_slices[pid_block];

    for (int h = 0; h < h_dim; ++h) {
        scalar_t sum = 0;
        for (int i = 0; i < in_dim; ++i) {
            scalar_t net_val = net_out[edge_idx * total_in_dim + in_start + i];
            scalar_t wm_val = wms[wm_start + h * in_dim + i];
            sum += net_val * wm_val;
        }
        H_out[edge_idx * total_h_dim + h_start + h] = sum;
    }
}

template <typename scalar_t>
__global__ void get_net_out_cuda_kernel(
    const scalar_t* __restrict__ H_in,
    const scalar_t* __restrict__ wms_H,
    scalar_t* __restrict__ net_out,
    const int* __restrict__ in_slices,
    const int* __restrict__ H_slices,
    const int* __restrict__ wm_H_slices,
    int num_edges,
    int total_in_dim,
    int total_h_dim) 
{
    int pid_block = blockIdx.y;
    int edge_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge_idx >= num_edges) return;

    int in_start = in_slices[pid_block];
    int in_dim = in_slices[pid_block + 1] - in_start;
    int h_start = H_slices[pid_block];
    int h_dim = H_slices[pid_block + 1] - h_start;
    int wm_h_start = wm_H_slices[pid_block];

    for (int i = 0; i < in_dim; ++i) {
        scalar_t sum = 0;
        for (int h = 0; h < h_dim; ++h) {
            scalar_t h_val = H_in[edge_idx * total_h_dim + h_start + h];
            scalar_t wm_h_val = wms_H[wm_h_start + i * h_dim + h];
            sum += h_val * wm_h_val;
        }
        net_out[edge_idx * total_in_dim + in_start + i] = sum;
    }
}

torch::Tensor launch_get_H(
    torch::Tensor net_out, torch::Tensor wms, 
    torch::Tensor in_slices, torch::Tensor H_slices, torch::Tensor wm_slices, 
    int num_blocks, int total_h_dim) 
{
    int num_edges = net_out.size(0);
    int total_in_dim = net_out.size(1);
    auto H_out = torch::zeros({num_edges, total_h_dim}, net_out.options());

    const int threads_per_block = 256;
    const int blocks_per_grid_x = (num_edges + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_per_grid_x, num_blocks);
    dim3 block(threads_per_block);

    AT_DISPATCH_FLOATING_TYPES(net_out.scalar_type(), "get_H_cuda", ([&] {
        get_H_cuda_kernel<scalar_t><<<grid, block>>>(
            net_out.data_ptr<scalar_t>(), wms.data_ptr<scalar_t>(),
            H_out.data_ptr<scalar_t>(),
            in_slices.data_ptr<int>(), H_slices.data_ptr<int>(), wm_slices.data_ptr<int>(),
            num_edges, total_in_dim, total_h_dim
        );
    }));
    return H_out;
}

torch::Tensor launch_get_net_out(
    torch::Tensor H_in, torch::Tensor wms_H, 
    torch::Tensor in_slices, torch::Tensor H_slices, torch::Tensor wm_H_slices, 
    int num_blocks, int total_in_dim) 
{
    int num_edges = H_in.size(0);
    int total_h_dim = H_in.size(1);
    auto net_out = torch::zeros({num_edges, total_in_dim}, H_in.options());

    const int threads_per_block = 256;
    const int blocks_per_grid_x = (num_edges + threads_per_block - 1) / threads_per_block;
    
    dim3 grid(blocks_per_grid_x, num_blocks);
    dim3 block(threads_per_block);

    AT_DISPATCH_FLOATING_TYPES(H_in.scalar_type(), "get_net_out_cuda", ([&] {
        get_net_out_cuda_kernel<scalar_t><<<grid, block>>>(
            H_in.data_ptr<scalar_t>(), wms_H.data_ptr<scalar_t>(),
            net_out.data_ptr<scalar_t>(),
            in_slices.data_ptr<int>(), H_slices.data_ptr<int>(), wm_H_slices.data_ptr<int>(),
            num_edges, total_in_dim, total_h_dim
        );
    }));
    return net_out;
}
'''
try:
    cuda_backend = torch.utils.cpp_extension.load_inline(
        name="tensor_decomp_cuda",
        cpp_sources="torch::Tensor launch_get_net_out(torch::Tensor H_in, torch::Tensor wms_H, torch::Tensor in_slices, torch::Tensor H_slices, torch::Tensor wm_H_slices, int num_blocks, int total_in_dim); torch::Tensor launch_get_H(torch::Tensor net_out, torch::Tensor wms, torch::Tensor in_slices, torch::Tensor H_slices, torch::Tensor wm_slices, int num_blocks, int total_h_dim);",
        cuda_sources=cuda_src,
        functions=['launch_get_H', 'launch_get_net_out'],
        with_cuda=True,
        verbose=False,
        extra_cflags=["-O3"]
    )
except Exception as e:
    print(f"Warning: Failed to compile CUDA inline extensions: {e}")
    cuda_backend = None

class CUDAGetHFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, net_out, decomp_obj):
        ctx.decomp_obj = decomp_obj
        return decomp_obj._get_H_impl(net_out)

    @staticmethod
    def backward(ctx, grad_output):
        grad_net_out = ctx.decomp_obj._get_adjoint(grad_output)
        return grad_net_out, None

class CudaE3TensorDecomp:
    def __init__(self, decomp_obj):
        self.device = decomp_obj.device
        self.dtype = decomp_obj.dtype
        self.num_blocks = len(decomp_obj.out_js_list)
        
        self.in_slices = torch.tensor(decomp_obj.in_slices, dtype=torch.int32, device=self.device)
        self.H_slices = torch.tensor(decomp_obj.H_slices, dtype=torch.int32, device=self.device)
        self.sort = decomp_obj.sort
        
        wms_flat_list = []
        wms_H_flat_list = []
        wms_adjoint_flat_list = []
        wm_slices = [0]
        wm_H_slices = [0]
        
        for i in range(self.num_blocks):
            in_dim = decomp_obj.in_slices[i+1] - decomp_obj.in_slices[i]
            h_dim = decomp_obj.H_slices[i+1] - decomp_obj.H_slices[i]
            
            wm = decomp_obj.wms[i].clone().view(h_dim, in_dim)
            wms_flat_list.append(wm.flatten())
            wm_slices.append(wm_slices[-1] + h_dim * in_dim)
            
            # Physical inverse (for get_net_out to match original semantics)
            wm_H_phys = decomp_obj.wms_H[i].clone().view(in_dim, h_dim).contiguous()
            wms_H_flat_list.append(wm_H_phys.flatten())
            wm_H_slices.append(wm_H_slices[-1] + in_dim * h_dim)

            # Mathematical adjoint (for backward pass autograd)
            wm_adjoint = wm.T.contiguous()
            wms_adjoint_flat_list.append(wm_adjoint.flatten())
            
        self.wms_flat = torch.cat(wms_flat_list).to(device=self.device, dtype=self.dtype)
        self.wm_slices = torch.tensor(wm_slices[:-1], dtype=torch.int32, device=self.device)
        
        self.wms_H_flat = torch.cat(wms_H_flat_list).to(device=self.device, dtype=self.dtype)
        self.wms_adjoint_flat = torch.cat(wms_adjoint_flat_list).to(device=self.device, dtype=self.dtype)
        self.wm_H_slices = torch.tensor(wm_H_slices[:-1], dtype=torch.int32, device=self.device)
        
        self.total_h_dim_val = int(decomp_obj.H_slices[-1])
        self.total_in_dim_val = int(decomp_obj.in_slices[-1])

    def get_H(self, net_out):
        return CUDAGetHFunction.apply(net_out, self)

    def _get_H_impl(self, net_out):
        if self.sort is not None:
            with torch.no_grad():
                net_out = self.sort.inverse(net_out)

        if cuda_backend is None:
            raise RuntimeError("CUDA extension failed to load. Please check PyTorch setup.")

        out = cuda_backend.launch_get_H(
            net_out.contiguous(), self.wms_flat.contiguous(), 
            self.in_slices, self.H_slices, self.wm_slices, 
            self.num_blocks, self.total_h_dim_val
        )
        return out

    def get_net_out(self, H):
        if cuda_backend is None:
            raise RuntimeError("CUDA extension failed to load. Please check PyTorch setup.")

        out = cuda_backend.launch_get_net_out(
            H.contiguous(), self.wms_H_flat.contiguous(), 
            self.in_slices, self.H_slices, self.wm_H_slices, 
            self.num_blocks, self.total_in_dim_val
        )
        if self.sort is not None:
            out = self.sort(out)
        return out

    def _get_adjoint(self, grad_output):
        if cuda_backend is None:
            raise RuntimeError("CUDA extension failed to load. Please check PyTorch setup.")

        out = cuda_backend.launch_get_net_out(
            grad_output.contiguous(), self.wms_adjoint_flat.contiguous(), 
            self.in_slices, self.H_slices, self.wm_H_slices, 
            self.num_blocks, self.total_in_dim_val
        )
        if self.sort is not None:
            out = self.sort(out)
        return out
