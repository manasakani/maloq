import pytest
import torch
import time
from fock_utils.utils_tensor_decomp import e3TensorDecomp, make_output_irreps
from fock_utils.cuda_backend import CudaE3TensorDecomp, cuda_backend

def test_cuda_backend():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        pytest.skip("CUDA not available; skipping CUDA backend test.")
        
    if cuda_backend is None:
        pytest.skip("CUDA extension failed to load; skipping CUDA backend test.")

    dtype = torch.float32
    print(f"Running test on: {device}\n")

    # 1. Mockup of a realistic orbital basis (H, C, O)
    orbital_basis = {
        1: [0],          
        6: [0, 0, 1],    
        8: [0, 0, 1]     
    }

    # 2. Init based on original logic
    targets, req_output_irreps, net_irreps_out, ls_list, out_js_list, out_slices, full_orb_interaction_list = make_output_irreps(orbital_basis)
    
    # 3. Initialization of the pure PyTorch class (baseline)
    decomp_torch = e3TensorDecomp(
        net_irreps_out=net_irreps_out, 
        out_js_list=out_js_list, 
        default_dtype_torch=dtype, 
        if_sort=True, 
        device_torch=device
    )

    # 4. Init of the new CUDA wrapper class
    decomp_cuda = CudaE3TensorDecomp(decomp_torch)

    num_edges = 15000  
    net_output_dim = decomp_torch.in_slices[-1] 
    H_dim = decomp_torch.H_slices[-1]

    print(f"--- TENSORS DIMENSIONS ---")
    print(f"Number of edges: {num_edges}")
    print(f"Dimension of network output (net_out): {net_output_dim}")
    print(f"Dimension of Fock matrix (H): {H_dim}\n")

    # 5. Generation of fake but realistic inputs (with gradient calls to test backward)
    dummy_net_out_torch = torch.randn((num_edges, net_output_dim), dtype=dtype, device=device, requires_grad=True)
    dummy_net_out_cuda = dummy_net_out_torch.detach().clone().requires_grad_(True)
    
    dummy_H_torch = torch.randn((num_edges, H_dim), dtype=dtype, device=device)
    dummy_H_cuda = dummy_H_torch.detach().clone()

    # --- FORWARD PASS TEST  (get_H) ---
    print("--- [1] FORWARD CHECK (get_H) ---")
    out_h_torch = decomp_torch.get_H(dummy_net_out_torch)
    out_h_cuda = decomp_cuda.get_H(dummy_net_out_cuda)
    
    match_forward = torch.allclose(out_h_torch, out_h_cuda, atol=1e-4, rtol=1e-4)
    max_diff_fwd = torch.max(torch.abs(out_h_torch - out_h_cuda)).item()
    print(f"(PyTorch == CUDA)? -> {'YES' if match_forward else 'NO'}")
    print(f"Maximum absolute difference: {max_diff_fwd:.2e}\n")

    assert match_forward, f"CUDA get_H mismatch vs PyTorch (max_abs_diff={max_diff_fwd:.2e})"

    # --- FORWARD PASS TEST  (get_net_out) ---
    print("--- [1.5] FORWARD CHECK (get_net_out) ---")
    out_net_torch = decomp_torch.get_net_out(dummy_H_torch)
    out_net_cuda = decomp_cuda.get_net_out(dummy_H_cuda)

    match_net = torch.allclose(out_net_torch, out_net_cuda, atol=1e-4, rtol=1e-4)
    max_diff_net = torch.max(torch.abs(out_net_torch - out_net_cuda)).item()
    print(f"(PyTorch == CUDA)? -> {'YES' if match_net else 'NO'}")
    print(f"Maximum absolute difference: {max_diff_net:.2e}\n")

    assert match_net, f"CUDA get_net_out mismatch vs PyTorch (max_abs_diff={max_diff_net:.2e})"

    # --- TEST BACKWARD PASS (get_H / autograd) ---
    print("--- [2] BACKWARD CHECK (Gradients of get_H) ---")
    
    grad_output = torch.randn_like(out_h_torch)
    loss_torch = (out_h_torch * grad_output).sum()
    loss_torch.backward()

    loss_cuda = (out_h_cuda * grad_output).sum()
    loss_cuda.backward()

    match_backward = torch.allclose(dummy_net_out_torch.grad, dummy_net_out_cuda.grad, atol=1e-4, rtol=1e-4)
    max_diff_bwd = torch.max(torch.abs(dummy_net_out_torch.grad - dummy_net_out_cuda.grad)).item()
    
    print(f"(PyTorch == CUDA)? -> {'YES' if match_backward else 'NO'}")
    print(f"Maximum absolute difference: {max_diff_bwd:.2e}\n")

    assert match_backward, f"CUDA get_H backward mismatch vs PyTorch (max_abs_diff={max_diff_bwd:.2e})"

    # --- BENCHMARK ---
    print("--- [3] MICRO-BENCHMARK for SPEEDUP ---")
    n_iters = 100
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = decomp_torch.get_H(dummy_net_out_torch)
            _ = decomp_cuda.get_H(dummy_net_out_cuda)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_iters):
            _ = decomp_torch.get_H(dummy_net_out_torch)
        torch.cuda.synchronize()
        time_torch = ((time.perf_counter() - start) / n_iters) * 1000

        start = time.perf_counter()
        for _ in range(n_iters):
            _ = decomp_cuda.get_H(dummy_net_out_cuda)
        torch.cuda.synchronize()
        time_cuda = ((time.perf_counter() - start) / n_iters) * 1000

    print(f"PyTorch time (get_H): {time_torch:.3f} ms / iter")
    print(f"CUDA kernel time (get_H): {time_cuda:.3f} ms / iter")
    print(f"CUDA speedup: {time_torch/time_cuda:.2f}x\n")
    print("Test completed successfully.")

if __name__ == "__main__":
    test_cuda_backend()