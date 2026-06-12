import torch
import time

import sys
from fock_utils.utils_tensor_decomp import e3TensorDecomp, make_output_irreps
import fock_utils.triton_backend as foo2
import pytest
from fock_utils import basis_sets as basis_sets_module

def test_triton_backend_get_H_matches_torch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping Triton backend test.")
    device = torch.device("cuda")
    dtype = torch.float32
    
    # Small mock basis to keep this test fast in CI
    orbital_basis = {1: [0], 6: [0, 0, 1], 8: [0, 0, 1]}
    targets, req_output_irreps, net_irreps_out, ls_list, out_js_list, out_slices, full_orb_interaction_list = make_output_irreps(orbital_basis)
    
    decomp = e3TensorDecomp(
        net_irreps_out=net_irreps_out,
        out_js_list=out_js_list,
        default_dtype_torch=dtype,
        if_sort=True,
        device_torch=device,
    )
    decomp_triton = foo2.TritonE3TensorDecomp(decomp)
    
    x = torch.randn((256, decomp.in_slices[-1]), dtype=dtype, device=device)
    # PyTorch baseline
    out_rows = 0
    out_cols = 0
    cols = []
    for i in range(len(decomp.out_js_list)):
        wms_shape = decomp.wms[i].shape
        out_rows += wms_shape[2]
        tmp = wms_shape[0] * wms_shape[1]
        out_cols += wms_shape[0] * wms_shape[1]
        cols.append(tmp)
    wms = torch.zeros((out_cols, out_rows), dtype=x.dtype, device=x.device)
    out_cols = 0
    for i in range(len(decomp.out_js_list)):
        rows = decomp.in_slices[i+1] - decomp.in_slices[i]
        in_slice = slice(decomp.in_slices[i], decomp.in_slices[i + 1])
        wms[out_cols:out_cols + cols[i], in_slice] = decomp.wms[i].reshape(-1, rows)
        out_cols += cols[i]
    y_ref = x @ wms.T
    
    # Triton backend
    y = decomp_triton.get_H(x)
    
    assert torch.allclose(y, y_ref, atol=1e-4, rtol=1e-4), "Triton mismatch vs PyTorch reference"

def _fmt_triton_config(cfg):
    if cfg is None:
        return "n/a"
    try:
        parts = []
        kwargs = getattr(cfg, "kwargs", None)
        if isinstance(kwargs, dict) and kwargs:
            kv = ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
            parts.append(kv)
        num_warps = getattr(cfg, "num_warps", None)
        if num_warps is not None:
            parts.append(f"num_warps={num_warps}")
        num_stages = getattr(cfg, "num_stages", None)
        if num_stages is not None:
            parts.append(f"num_stages={num_stages}")
        if parts:
            return "{" + ", ".join(parts) + "}"
    except Exception:
        pass
    return str(cfg)


def _print_autotune_best(module_obj, module_name, kernel_names, max_entries=4):
    if module_obj is None:
        return
    print(f"\n[Autotune best/cached configs: {module_name}]")
    for kname in kernel_names:
        kernel = getattr(module_obj, kname, None)
        if kernel is None:
            continue

        best = getattr(kernel, "best_config", None)
        cache = getattr(kernel, "cache", None)

        if best is not None:
            print(f"  {kname}: best_config={_fmt_triton_config(best)}")

        if isinstance(cache, dict) and len(cache) > 0:
            print(f"  {kname}: cached_keys={len(cache)}")
            shown = 0
            for key, value in cache.items():
                cfg = value
                if isinstance(value, tuple):
                    found = None
                    for item in value:
                        if hasattr(item, "kwargs") or hasattr(item, "num_warps"):
                            found = item
                            break
                    if found is not None:
                        cfg = found
                print(f"    key={key} -> {_fmt_triton_config(cfg)}")
                shown += 1
                if shown >= max_entries:
                    break
        elif best is None:
            print(f"  {kname}: no autotune cache populated yet")


def _print_aligned_table(headers, rows, right_align=None):
    if right_align is None:
        right_align = [False] * len(headers)

    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def _fmt_row(values):
        parts = []
        for i, cell in enumerate(values):
            text = str(cell)
            if right_align[i]:
                parts.append(text.rjust(widths[i]))
            else:
                parts.append(text.ljust(widths[i]))
        return " | ".join(parts)

    print(_fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(_fmt_row(row))

def get_H(self, net_out): 
    # torch.cuda.synchronize() 
    # nvtx.range_push("get_H")

    r''' get openmx type H from net output '''

    if self.sort is not None:
        net_out = self.sort.inverse(net_out)
    # out = []
    out_rows = 0
    out_cols = 0
    cols = []
    for i in range(len(self.out_js_list)):
        wms_shape = self.wms[i].shape
        out_rows += wms_shape[2]
        tmp = wms_shape[0] * wms_shape[1]
        out_cols += wms_shape[0] * wms_shape[1]
        cols.append(tmp)
    wms = torch.zeros((out_cols, out_rows), dtype=net_out.dtype, device=net_out.device)
    out_cols = 0
    useful_flop = 0
    for i in range(len(self.out_js_list)):
        rows = self.in_slices[i+1] - self.in_slices[i]
        in_slice = slice(self.in_slices[i], self.in_slices[i + 1])
        wms[out_cols:out_cols + cols[i], in_slice] = self.wms[i].reshape(-1, rows)
        out_cols += cols[i]
        useful_flop += 2 * net_out.shape[0] * rows * cols[i] 
    total_flop = 2 * net_out.shape[0] * out_rows * out_cols
    
    return net_out @ wms.T, total_flop, useful_flop


def get_H_original(self, net_out):
    r''' get openmx type H from net output (original utils_tensor_decomp implementation) '''

    if self.sort is not None:
        net_out = self.sort.inverse(net_out)
    out = []

    for i in range(len(self.out_js_list)):
        in_slice = slice(self.in_slices[i], self.in_slices[i + 1])
        net_out_block = net_out[:, in_slice]
        H_block = torch.sum(self.wms[i][None, :, :, :] * net_out_block[:, None, None, :], dim=-1)
        out.append(H_block.reshape(net_out.shape[0], -1))

    return torch.cat(out, dim=-1)


def get_H_2(self, net_out):
    if self.sort is not None:
        net_out = self.sort.inverse(net_out)

    out_cols = 0
    cols = []
    for i in range(len(self.out_js_list)):
        wms_shape = self.wms[i].shape
        tmp = wms_shape[0] * wms_shape[1]
        out_cols += wms_shape[0] * wms_shape[1]
        cols.append(tmp)
    out = torch.empty((net_out.shape[0], out_cols), dtype=net_out.dtype, device=net_out.device)
    offset = 0
    for i in range(len(self.out_js_list)):
        in_slice = slice(self.in_slices[i], self.in_slices[i + 1])
        net_out_block = net_out[:, in_slice]
        # H_block = torch.sum(self.wms[i][None, :, :, :] * net_out_block[:, None, None, :], dim=-1)
        wms_shape = self.wms[i].shape
        cols = wms_shape[0] * wms_shape[1]
        wms = self.wms[i].reshape(cols, wms_shape[2])
        H_block = net_out_block @ wms.T
        # out.append(H_block.reshape(net_out.shape[0], -1))
        out[:, offset:offset+cols] = H_block
        offset += cols
    return out

    # out = torch.empty((net_out.shape[0], out_cols), dtype=net_out.dtype, device=net_out.device)
    # offset = 0
    # for i in range(len(self.out_js_list)):
    #     in_slice = slice(self.in_slices[i], self.in_slices[i + 1])
    #     net_out_block = net_out[:, in_slice]
    #     # H_block = torch.sum(self.wms[i][None, :, :, :] * net_out_block[:, None, None, :], dim=-1)
    #     wms_shape = self.wms[i].shape
    #     cols = wms_shape[0] * wms_shape[1]
    #     wms = self.wms[i].reshape(cols, wms_shape[2])
    #     H_block = net_out_block @ wms.T
    #     # out.append(H_block.reshape(net_out.shape[0], -1))
    #     out[:, offset:offset+cols] = H_block
    #     offset += cols
    # return out
    # torch.cuda.synchronize()
    # nvtx.range_pop()
    # return torch.cat(out, dim=-1) # output shape: [edge, (4 spin components,) H_flattened_concatenated]

def run_benchmark_for_basis(basis_name, orbital_basis, device, dtype, num_edges=100000, n_iters=100, n_warmup=10):
    """Run the full benchmark flow for a single basis set."""
    print(f"\n{'='*80}")
    print(f"Benchmarking basis set: {basis_name}")
    print(f"{'='*80}")
    
    # 2. Generate necessary lists for initialization based on the original logic
    targets, req_output_irreps, net_irreps_out, ls_list, out_js_list, out_slices, full_orb_interaction_list = make_output_irreps(orbital_basis)
    
    # 3. Initialize the Decomp class as it is done in splittrainer
    decomp = e3TensorDecomp(
        net_irreps_out=net_irreps_out, 
        out_js_list=out_js_list, 
        default_dtype_torch=dtype, 
        if_sort=True, 
        device_torch=device
    )

    # Initialize the Triton wrapper classes
    decomp_triton = foo2.TritonE3TensorDecomp(decomp)
    decomp_triton_l2 = None
    if foo2 is not None and hasattr(foo2, "TritonE3TensorDecompL2"):
        decomp_triton_l2 = foo2.TritonE3TensorDecompL2(decomp)

    # The compressed tensor generated by the GNN (network output)
    net_output_dim = decomp.in_slices[-1] 
    
    # The physical "decompressed" Fock tensor (H matrix)
    H_dim = decomp.H_slices[-1]

    print(f"\n--- TENSOR SHAPES ---")
    print(f"Number of edges : {num_edges}")
    print(f"Net Out Dim     : {net_output_dim} -> Shape: ({num_edges}, {net_output_dim})")
    print(f"H Matrix Dim    : {H_dim} -> Shape: ({num_edges}, {H_dim})")
    print(f"wms blocks      : {len(decomp.out_js_list)}")

    # 4. GPU Memory Allocation
    torch.manual_seed(42)  # Fix seed for reproducibility
    dummy_net_out = torch.randn((num_edges, net_output_dim), dtype=dtype, device=device)
    dummy_H = torch.randn((num_edges, H_dim), dtype=dtype, device=device)

    # 5. Warm-up (to avoid measuring lazy JIT allocations)
    print("\n[Warming up GPU...]")
    for _ in range(n_warmup):
        _, total_flop, useful_flop = get_H(decomp, dummy_net_out)
    torch.cuda.synchronize()

    # 6. BENCHMARK: get_H
    print(f"\n[Benchmarking get_H over {n_iters} iterations...]")
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        out_h, _, _ = get_H(decomp, dummy_net_out)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    import statistics
    get_H_time_ms = statistics.median(times)
    print(f"--> get_H median time: {get_H_time_ms:.3f} ms")

    # Benchmark: get_H_original (PyTorch original implementation)
    print(f"\n[Benchmarking get_H_original (PyTorch) over {n_iters} iterations...]")
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        out_h_original = get_H_original(decomp, dummy_net_out)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    get_H_original_time_ms = statistics.median(times)
    print(f"--> get_H_original (PyTorch) median time: {get_H_original_time_ms:.3f} ms")
    print(f"    Speedup of get_H over get_H_original: {get_H_original_time_ms / get_H_time_ms:.2f}x")
    match_original_ref = torch.allclose(out_h, out_h_original, atol=1e-5, rtol=1e-5)
    rel_error_original_ref = torch.norm(out_h - out_h_original) / torch.norm(out_h)

    # Benchmark: get_H_2 (PyTorch)
    print(f"\n[Benchmarking get_H_2 (PyTorch) over {n_iters} iterations...]")
    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        out_h_2 = get_H_2(decomp, dummy_net_out)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    import statistics
    get_H_2_time_ms = statistics.median(times)
    print(f"--> get_H_2 (PyTorch) median time: {get_H_2_time_ms:.3f} ms")
    print(f"    Speedup of get_H_2 over get_H_original: {get_H_original_time_ms / get_H_2_time_ms:.2f}x")
    match_h2_ref = torch.allclose(out_h, out_h_2, atol=1e-5, rtol=1e-5)
    rel_error_h2_ref = torch.norm(out_h - out_h_2) / torch.norm(out_h)



    # 6.5 BENCHMARK: get_H (TRITON)
    print(f"\n[Benchmarking get_H (Triton naive) over {n_iters} iterations...]")
    for _ in range(n_warmup): # warmup triton specific JIT compiling
        _ = decomp_triton.get_H(dummy_net_out)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        out_h_triton = decomp_triton.get_H(dummy_net_out)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    import statistics
    get_H_triton_time_ms = statistics.median(times)
    print(f"--> get_H (Triton) median time: {get_H_triton_time_ms:.3f} ms")
    print(f"    Speedup over get_H_original: {get_H_original_time_ms / get_H_triton_time_ms:.2f}x")

    # Sanity Check to confirm numerics match!
    match_triton_ref = torch.allclose(out_h, out_h_triton, atol=1e-5, rtol=1e-5)
    print(f"\n[Sanity Check] Do Triton and PyTorch provide the identical H matrix? -> {'YES' if match_triton_ref else 'NO!!!'}")
    rel_error = torch.norm(out_h - out_h_triton) / torch.norm(out_h)
    print(f"Relative error between PyTorch and Triton outputs: {rel_error:.2e}")

    match_h2_triton = torch.allclose(out_h_2, out_h_triton, atol=1e-5, rtol=1e-5)
    print(f"\n[Sanity Check] Do get_H_2 (PyTorch) and Triton provide the identical H matrix? -> {'YES' if match_h2_triton else 'NO!!!'}")
    rel_error_2 = torch.norm(out_h_2 - out_h_triton) / torch.norm(out_h_2)
    print(f"Relative error between get_H_2 (PyTorch) and Triton outputs: {rel_error_2:.2e}")

    get_H_triton_l2_time_ms = None
    match_triton_l2_ref = None
    rel_error_l2 = None
    if decomp_triton_l2 is not None:
        print(f"\n[Benchmarking get_H (Triton L2) over {n_iters} iterations...]")
        for _ in range(n_warmup):
            _ = decomp_triton_l2.get_H(dummy_net_out)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            out_h_triton_l2 = decomp_triton_l2.get_H(dummy_net_out)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        import statistics
        get_H_triton_l2_time_ms = statistics.median(times)
        print(f"--> get_H (Triton L2) median time: {get_H_triton_l2_time_ms:.3f} ms")
        print(f"    Speedup over get_H_original: {get_H_original_time_ms / get_H_triton_l2_time_ms:.2f}x")
        print(f"    Speedup over Triton naive: {get_H_triton_time_ms / get_H_triton_l2_time_ms:.2f}x")

        match_triton_l2_ref = torch.allclose(out_h, out_h_triton_l2, atol=1e-5, rtol=1e-5)
        rel_error_l2 = torch.norm(out_h - out_h_triton_l2) / torch.norm(out_h)
        print(
            "[Sanity Check] Do Triton L2 and PyTorch provide the identical H matrix? "
            f"-> {'YES' if match_triton_l2_ref else 'NO!!!'}"
        )
        print(f"Relative error between PyTorch and Triton L2 outputs: {rel_error_l2:.2e}")

    impl_results = [
        {
            "basis": basis_name,
            "name": "PyTorch get_H_original",
            "time_ms": get_H_original_time_ms,
            "speedup_vs_original": 1.0,
            "match": bool(match_original_ref),
            "rel_error": rel_error_original_ref.item(),
            "tflops": useful_flop / (get_H_original_time_ms / 1000.0) / 1e12,
        },
        {
            "basis": basis_name,
            "name": "PyTorch get_H",
            "time_ms": get_H_time_ms,
            "speedup_vs_original": get_H_original_time_ms / get_H_time_ms,
            "match": True,
            "rel_error": 0.0,
            "tflops": useful_flop / (get_H_time_ms / 1000.0) / 1e12,
        },
        {
            "basis": basis_name,
            "name": "PyTorch get_H_2",
            "time_ms": get_H_2_time_ms,
            "speedup_vs_original": get_H_original_time_ms / get_H_2_time_ms,
            "match": bool(match_h2_ref),
            "rel_error": rel_error_h2_ref.item(),
            "tflops": useful_flop / (get_H_2_time_ms / 1000.0) / 1e12,
        },

        {
            "basis": basis_name,
            "name": "Triton naive",
            "time_ms": get_H_triton_time_ms,
            "speedup_vs_original": get_H_original_time_ms / get_H_triton_time_ms,
            "match": bool(match_triton_ref),
            "rel_error": rel_error.item(),
            "tflops": useful_flop / (get_H_triton_time_ms / 1000.0) / 1e12,
        },
    ]

    if get_H_triton_l2_time_ms is not None:
        impl_results.append(
            {
                "basis": basis_name,
                "name": "Triton naive L2",
                "time_ms": get_H_triton_l2_time_ms,
                "speedup_vs_original": get_H_original_time_ms / get_H_triton_l2_time_ms,
                "match": bool(match_triton_l2_ref),
                "rel_error": rel_error_l2.item(),
                "tflops": useful_flop / (get_H_triton_l2_time_ms / 1000.0) / 1e12,
            }
        )

    grouped_results = []
    for grouping_size in [8, 16, 32, 64]:
        decomp_triton_grouped = foo2.TritonE3TensorDecomp(
            decomp,
            grouped_max_in_dim=grouping_size,
            grouped_max_h_dim=grouping_size,
            use_grouped_kernel=True,
        )
        decomp_triton_grouped.print_group_debug_info(max_groups=20)

        print(
            f"\n[Benchmarking get_H (Triton Grouped, size={grouping_size}) over {n_iters} iterations...]"
        )
        for _ in range(n_warmup):  # warmup triton specific JIT compiling
            _ = decomp_triton_grouped.get_H(dummy_net_out)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            out_h_triton_grouped = decomp_triton_grouped.get_H(dummy_net_out)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        import statistics
        get_H_triton_grouped_time_ms = statistics.median(times)
        print(
            f"--> get_H (Triton Grouped, size={grouping_size}) median time: "
            f"{get_H_triton_grouped_time_ms:.3f} ms"
        )
        print(
            "    Speedup of Triton Grouped over get_H_original: "
            f"{get_H_original_time_ms / get_H_triton_grouped_time_ms:.2f}x"
        )
        print(
            "    Speedup of Triton Grouped over Triton Naive: "
            f"{get_H_triton_time_ms / get_H_triton_grouped_time_ms:.2f}x"
        )

        match = torch.allclose(out_h, out_h_triton_grouped, atol=1e-5, rtol=1e-5)
        print(
            "\n[Sanity Check] Do Triton Grouped and PyTorch provide the identical H matrix? "
            f"-> {'YES' if match else 'NO!!!'}"
        )
        rel_error_grouped = torch.norm(out_h - out_h_triton_grouped) / torch.norm(out_h)
        print(f"Relative error between PyTorch and Triton Grouped outputs: {rel_error_grouped:.2e}")

        grouped_results.append(
            {
                "basis": basis_name,
                "name": f"Triton grouped (size={grouping_size})",
                "size": grouping_size,
                "time_ms": get_H_triton_grouped_time_ms,
                "vs_original": get_H_original_time_ms / get_H_triton_grouped_time_ms,
                "vs_naive": get_H_triton_time_ms / get_H_triton_grouped_time_ms,
                "match": match,
                "rel_error": rel_error_grouped.item(),
                "imb_max_mean": float(getattr(decomp_triton_grouped, "group_imbalance", {}).get("useful_flop_imbalance_ratio", float("nan"))),
                "imb_max_min": float(getattr(decomp_triton_grouped, "group_imbalance", {}).get("useful_flop_max_over_min", float("nan"))),
                "tflops": useful_flop / (get_H_triton_grouped_time_ms / 1000.0) / 1e12,
            }
        )

        if foo2_l2 is not None and hasattr(foo2_l2, "TritonE3TensorDecompL2"):
            decomp_triton_grouped_l2 = foo2_l2.TritonE3TensorDecompL2(
                decomp,
                grouped_max_in_dim=grouping_size,
                grouped_max_h_dim=grouping_size,
                use_grouped_kernel=True,
            )
            decomp_triton_grouped_l2.print_group_debug_info(max_groups=20)

            print(
                f"\n[Benchmarking get_H (Triton Grouped L2, size={grouping_size}) over {n_iters} iterations...]"
            )
            for _ in range(n_warmup):
                _ = decomp_triton_grouped_l2.get_H(dummy_net_out)
            torch.cuda.synchronize()

            times = []
            for _ in range(n_iters):
                start = time.perf_counter()
                out_h_triton_grouped_l2 = decomp_triton_grouped_l2.get_H(dummy_net_out)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
            import statistics
            get_H_triton_grouped_l2_time_ms = statistics.median(times)
            print(
                f"--> get_H (Triton Grouped L2, size={grouping_size}) median time: "
                f"{get_H_triton_grouped_l2_time_ms:.3f} ms"
            )
            print(
                "    Speedup of Triton Grouped L2 over get_H_original: "
                f"{get_H_original_time_ms / get_H_triton_grouped_l2_time_ms:.2f}x"
            )
            print(
                "    Speedup of Triton Grouped L2 over Triton Naive: "
                f"{get_H_triton_time_ms / get_H_triton_grouped_l2_time_ms:.2f}x"
            )

            match_l2 = torch.allclose(out_h, out_h_triton_grouped_l2, atol=1e-5, rtol=1e-5)
            rel_error_grouped_l2 = torch.norm(out_h - out_h_triton_grouped_l2) / torch.norm(out_h)
            print(
                "\n[Sanity Check] Do Triton Grouped L2 and PyTorch provide the identical H matrix? "
                f"-> {'YES' if match_l2 else 'NO!!!'}"
            )
            print(
                "Relative error between PyTorch and Triton Grouped L2 outputs: "
                f"{rel_error_grouped_l2:.2e}"
            )

            grouped_results.append(
                {
                    "basis": basis_name,
                    "name": f"Triton grouped L2 (size={grouping_size})",
                    "size": grouping_size,
                    "time_ms": get_H_triton_grouped_l2_time_ms,
                    "vs_original": get_H_original_time_ms / get_H_triton_grouped_l2_time_ms,
                    "vs_naive": get_H_triton_time_ms / get_H_triton_grouped_l2_time_ms,
                    "match": match_l2,
                    "rel_error": rel_error_grouped_l2.item(),
                    "imb_max_mean": float(getattr(decomp_triton_grouped_l2, "group_imbalance", {}).get("useful_flop_imbalance_ratio", float("nan"))),
                    "imb_max_min": float(getattr(decomp_triton_grouped_l2, "group_imbalance", {}).get("useful_flop_max_over_min", float("nan"))),
                    "tflops": useful_flop / (get_H_triton_grouped_l2_time_ms / 1000.0) / 1e12,
                }
            )

        if hasattr(foo2, "BalancedTritonE3TensorDecomp"):
            decomp_triton_grouped_balanced = foo2.BalancedTritonE3TensorDecomp(
                decomp,
                grouped_max_in_dim=grouping_size,
                grouped_max_h_dim=grouping_size,
                use_grouped_kernel=True,
            )
            decomp_triton_grouped_balanced.print_group_debug_info(max_groups=20)

            # A/B comparison of group-load imbalance for sequential vs balanced grouping.
            seq_imb = getattr(decomp_triton_grouped, "group_imbalance", {})
            bal_imb = getattr(decomp_triton_grouped_balanced, "group_imbalance", {})
            if seq_imb and bal_imb:
                seq_ratio = float(seq_imb.get("useful_flop_imbalance_ratio", float("nan")))
                bal_ratio = float(bal_imb.get("useful_flop_imbalance_ratio", float("nan")))
                seq_spread = float(seq_imb.get("useful_flop_max_over_min", float("nan")))
                bal_spread = float(bal_imb.get("useful_flop_max_over_min", float("nan")))
                seq_groups = int(seq_imb.get("num_groups", -1))
                bal_groups = int(bal_imb.get("num_groups", -1))

                print("\n[Group Imbalance A/B Comparison]")
                print(
                    f"  {'metric':>28} | {'sequential':>12} | {'balanced':>12} | {'improvement':>12}"
                )
                print("  " + "-" * 74)
                print(
                    f"  {'num_groups':>28} | {seq_groups:12d} | {bal_groups:12d} | {'n/a':>12}"
                )
                print(
                    f"  {'max/mean useful_flop':>28} | {seq_ratio:12.3f} | {bal_ratio:12.3f} | "
                    f"{(seq_ratio / bal_ratio):12.3f}x"
                    if bal_ratio > 0
                    else f"  {'max/mean useful_flop':>28} | {seq_ratio:12.3f} | {bal_ratio:12.3f} | {'n/a':>12}"
                )
                print(
                    f"  {'max/min useful_flop':>28} | {seq_spread:12.3f} | {bal_spread:12.3f} | "
                    f"{(seq_spread / bal_spread):12.3f}x"
                    if bal_spread > 0
                    else f"  {'max/min useful_flop':>28} | {seq_spread:12.3f} | {bal_spread:12.3f} | {'n/a':>12}"
                )

            print(
                f"\n[Benchmarking get_H (Triton Grouped Balanced, size={grouping_size}) over {n_iters} iterations...]"
            )
            for _ in range(n_warmup):
                _ = decomp_triton_grouped_balanced.get_H(dummy_net_out)
            torch.cuda.synchronize()

            times = []
            for _ in range(n_iters):
                start = time.perf_counter()
                out_h_triton_grouped_balanced = decomp_triton_grouped_balanced.get_H(dummy_net_out)
                torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
            import statistics
            get_H_triton_grouped_balanced_time_ms = statistics.median(times)
            print(
                f"--> get_H (Triton Grouped Balanced, size={grouping_size}) median time: "
                f"{get_H_triton_grouped_balanced_time_ms:.3f} ms"
            )
            print(
                "    Speedup of Triton Grouped Balanced over get_H_original: "
                f"{get_H_original_time_ms / get_H_triton_grouped_balanced_time_ms:.2f}x"
            )
            print(
                "    Speedup of Triton Grouped Balanced over Triton Naive: "
                f"{get_H_triton_time_ms / get_H_triton_grouped_balanced_time_ms:.2f}x"
            )

            match_balanced = torch.allclose(out_h, out_h_triton_grouped_balanced, atol=1e-5, rtol=1e-5)
            print(
                "\n[Sanity Check] Do Triton Grouped Balanced and PyTorch provide the identical H matrix? "
                f"-> {'YES' if match_balanced else 'NO!!!'}"
            )
            rel_error_balanced = torch.norm(out_h - out_h_triton_grouped_balanced) / torch.norm(out_h)
            print(
                "Relative error between PyTorch and Triton Grouped Balanced outputs: "
                f"{rel_error_balanced:.2e}"
            )

            grouped_results.append(
                {
                    "basis": basis_name,
                    "name": f"Triton grouped balanced (size={grouping_size})",
                    "size": grouping_size,
                    "time_ms": get_H_triton_grouped_balanced_time_ms,
                    "vs_original": get_H_original_time_ms / get_H_triton_grouped_balanced_time_ms,
                    "vs_naive": get_H_triton_time_ms / get_H_triton_grouped_balanced_time_ms,
                    "match": match_balanced,
                    "rel_error": rel_error_balanced.item(),
                    "imb_max_mean": float(getattr(decomp_triton_grouped_balanced, "group_imbalance", {}).get("useful_flop_imbalance_ratio", float("nan"))),
                    "imb_max_min": float(getattr(decomp_triton_grouped_balanced, "group_imbalance", {}).get("useful_flop_max_over_min", float("nan"))),
                    "tflops": useful_flop / (get_H_triton_grouped_balanced_time_ms / 1000.0) / 1e12,
                }
            )

            if foo2_l2 is not None and hasattr(foo2_l2, "BalancedTritonE3TensorDecompL2"):
                decomp_triton_grouped_balanced_l2 = foo2_l2.BalancedTritonE3TensorDecompL2(
                    decomp,
                    grouped_max_in_dim=grouping_size,
                    grouped_max_h_dim=grouping_size,
                    use_grouped_kernel=True,
                )
                decomp_triton_grouped_balanced_l2.print_group_debug_info(max_groups=20)

                print(
                    f"\n[Benchmarking get_H (Triton Grouped Balanced L2, size={grouping_size}) over {n_iters} iterations...]"
                )
                for _ in range(n_warmup):
                    _ = decomp_triton_grouped_balanced_l2.get_H(dummy_net_out)
                torch.cuda.synchronize()

                times = []
                for _ in range(n_iters):
                    start = time.perf_counter()
                    out_h_triton_grouped_balanced_l2 = decomp_triton_grouped_balanced_l2.get_H(dummy_net_out)
                    torch.cuda.synchronize()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)
                import statistics
                get_H_triton_grouped_balanced_l2_time_ms = statistics.median(times)
                print(
                    f"--> get_H (Triton Grouped Balanced L2, size={grouping_size}) median time: "
                    f"{get_H_triton_grouped_balanced_l2_time_ms:.3f} ms"
                )
                print(
                    "    Speedup of Triton Grouped Balanced L2 over get_H_original: "
                    f"{get_H_original_time_ms / get_H_triton_grouped_balanced_l2_time_ms:.2f}x"
                )
                print(
                    "    Speedup of Triton Grouped Balanced L2 over Triton Naive: "
                    f"{get_H_triton_time_ms / get_H_triton_grouped_balanced_l2_time_ms:.2f}x"
                )

                match_balanced_l2 = torch.allclose(out_h, out_h_triton_grouped_balanced_l2, atol=1e-5, rtol=1e-5)
                rel_error_balanced_l2 = torch.norm(out_h - out_h_triton_grouped_balanced_l2) / torch.norm(out_h)
                print(
                    "\n[Sanity Check] Do Triton Grouped Balanced L2 and PyTorch provide the identical H matrix? "
                    f"-> {'YES' if match_balanced_l2 else 'NO!!!'}"
                )
                print(
                    "Relative error between PyTorch and Triton Grouped Balanced L2 outputs: "
                    f"{rel_error_balanced_l2:.2e}"
                )

                grouped_results.append(
                    {
                        "basis": basis_name,
                        "name": f"Triton grouped balanced L2 (size={grouping_size})",
                        "size": grouping_size,
                        "time_ms": get_H_triton_grouped_balanced_l2_time_ms,
                        "vs_original": get_H_original_time_ms / get_H_triton_grouped_balanced_l2_time_ms,
                        "vs_naive": get_H_triton_time_ms / get_H_triton_grouped_balanced_l2_time_ms,
                        "match": match_balanced_l2,
                        "rel_error": rel_error_balanced_l2.item(),
                        "imb_max_mean": float(getattr(decomp_triton_grouped_balanced_l2, "group_imbalance", {}).get("useful_flop_imbalance_ratio", float("nan"))),
                        "imb_max_min": float(getattr(decomp_triton_grouped_balanced_l2, "group_imbalance", {}).get("useful_flop_max_over_min", float("nan"))),
                        "tflops": useful_flop / (get_H_triton_grouped_balanced_l2_time_ms / 1000.0) / 1e12,
                    }
                )

    print("\n=== Summary for this Basis Set ===")
    headers = [
        "implementation",
        "time_ms",
        "tflops",
        "speedup_vs_original",
        "match",
        "rel_error",
        "imb_max_mean",
        "imb_max_min",
    ]
    rows = []
    for r in impl_results:
        imb_max_mean = r.get("imb_max_mean", float("nan"))
        imb_max_min = r.get("imb_max_min", float("nan"))
        imb_max_mean_str = f"{imb_max_mean:12.3f}" if imb_max_mean == imb_max_mean else f"{'n/a':>12}"
        imb_max_min_str = f"{imb_max_min:11.3f}" if imb_max_min == imb_max_min else f"{'n/a':>11}"
        tflops_str = f"{r.get('tflops', 0.0):.3f}"
        rows.append([
            r['name'],
            f"{r['time_ms']:.3f}",
            tflops_str,
            f"{r['speedup_vs_original']:.2f}",
            str(r['match']),
            f"{r['rel_error']:.2e}",
            imb_max_mean_str.strip(),
            imb_max_min_str.strip(),
        ])
    for r in grouped_results:
        imb_max_mean = r.get("imb_max_mean", float("nan"))
        imb_max_min = r.get("imb_max_min", float("nan"))
        imb_max_mean_str = f"{imb_max_mean:12.3f}" if imb_max_mean == imb_max_mean else f"{'n/a':>12}"
        imb_max_min_str = f"{imb_max_min:11.3f}" if imb_max_min == imb_max_min else f"{'n/a':>11}"
        tflops_str = f"{r.get('tflops', 0.0):.3f}"
        rows.append([
            r['name'],
            f"{r['time_ms']:.3f}",
            tflops_str,
            f"{r['vs_original']:.2f}",
            str(r['match']),
            f"{r['rel_error']:.2e}",
            imb_max_mean_str.strip(),
            imb_max_min_str.strip(),
        ])
    _print_aligned_table(
        headers,
        rows,
        right_align=[False, True, True, True, True, True, True, True],
    )

    # Compact L2 uplift summary against non-L2 counterparts.
    all_rows = impl_results + grouped_results
    time_by_name = {r["name"]: r["time_ms"] for r in all_rows}
    l2_pairs = []
    for name in time_by_name:
        if " L2 (" in name:
            base = name.replace(" L2 (", " (")
        elif name.endswith(" L2"):
            base = name[:-3]
        else:
            continue
        if base in time_by_name:
            l2_pairs.append((base, name))

    if l2_pairs:
        print("\n=== L2 Uplift Summary ===")
        print(f"{'base':>34} | {'l2':>34} | {'base_ms':>10} | {'l2_ms':>10} | {'uplift':>8}")
        print("-" * 110)
        for base, l2 in sorted(l2_pairs):
            base_ms = time_by_name[base]
            l2_ms = time_by_name[l2]
            uplift = (base_ms / l2_ms) if l2_ms > 0 else float("inf")
            print(f"{base:>34} | {l2:>34} | {base_ms:10.3f} | {l2_ms:10.3f} | {uplift:8.3f}x")

    # Print Triton autotune selections observed for this basis run.
    _print_autotune_best(
        foo2,
        "triton_tensor_decomp",
        ["get_H_triton_block_kernel", "get_H_triton_grouped_kernel"],
    )
    _print_autotune_best(
        foo2,
        "triton_tensor_decomp_l2_final",
        ["get_H_triton_block_kernel_l2", "get_H_triton_grouped_kernel_l2"],
    )

    # ==========================================
    # BENCHMARK: get_net_out
    # ==========================================
    print("\n" + "="*80)
    print("Benchmarking get_net_out".center(80))
    print("="*80)
    net_out_results = []

    print(f"\n[Benchmarking get_net_out (PyTorch) over {n_iters} iterations...]")
    for _ in range(n_warmup):
        _ = decomp.get_net_out(dummy_H)
    torch.cuda.synchronize()

    times = []
    for _ in range(n_iters):
        start = time.perf_counter()
        out_net_original = decomp.get_net_out(dummy_H)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)
    import statistics
    get_net_out_original_time_ms = statistics.median(times)
    print(f"--> get_net_out (PyTorch) median time: {get_net_out_original_time_ms:.3f} ms")
    net_out_results.append({
        "name": "PyTorch get_net_out",
        "basis": basis_name,
        "time_ms": get_net_out_original_time_ms,
        "speedup_vs_original": 1.0,
        "match": True
    })


    if decomp_triton_l2 is not None and hasattr(decomp_triton_l2, 'get_net_out_groupsweep'):
        print(f"\n[Benchmarking get_net_out_groupsweep (Triton L2) over {n_iters} iterations...]")
        for _ in range(n_warmup):
            _ = decomp_triton_l2.get_net_out_groupsweep(dummy_H, group_size=3)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            out_net_triton_l2_gs = decomp_triton_l2.get_net_out_groupsweep(dummy_H, group_size=3)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        import statistics
        get_net_out_triton_l2_gs_time_ms = statistics.median(times)
        print(f"--> get_net_out_groupsweep (Triton L2) median time: {get_net_out_triton_l2_gs_time_ms:.3f} ms")
        print(f"    Speedup over get_net_out (PyTorch): {get_net_out_original_time_ms / get_net_out_triton_l2_gs_time_ms:.2f}x")

        match_triton_l2_gs_ref = torch.allclose(out_net_original, out_net_triton_l2_gs, atol=1e-3, rtol=1e-3)
        rel_error_l2_gs = torch.norm(out_net_original - out_net_triton_l2_gs) / (torch.norm(out_net_original) + 1e-6)
        print(
            "[Sanity Check] Do Triton L2 groupsweep and PyTorch provide the identical net_out tensor? "
            f"-> {'YES' if match_triton_l2_gs_ref else 'NO!!!'}"
        )
        print(f"Relative error between PyTorch and Triton L2 groupsweep outputs: {rel_error_l2_gs:.2e}")
        
        net_out_results.append({
            "name": "Triton grouped L2 get_net_out",
            "basis": basis_name,
            "time_ms": get_net_out_triton_l2_gs_time_ms,
            "speedup_vs_original": get_net_out_original_time_ms / get_net_out_triton_l2_gs_time_ms,
            "match": match_triton_l2_gs_ref
        })
        
    try:
        _print_autotune_best(
            foo2,
            "triton_tensor_decomp_l2_final",
            ["get_net_out_triton_grouped_kernel_l2"],
        )
    except Exception:
        pass

    if decomp_triton_l2 is not None and hasattr(decomp_triton_l2, 'get_net_out'):
        print(f"\n[Benchmarking get_net_out (Triton L2) over {n_iters} iterations...]")
        for _ in range(n_warmup):
            _ = decomp_triton_l2.get_net_out(dummy_H)
        torch.cuda.synchronize()

        times = []
        for _ in range(n_iters):
            start = time.perf_counter()
            out_net_triton_l2 = decomp_triton_l2.get_net_out(dummy_H)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)
        import statistics
        get_net_out_triton_l2_time_ms = statistics.median(times)
        print(f"--> get_net_out (Triton L2) median time: {get_net_out_triton_l2_time_ms:.3f} ms")
        print(f"    Speedup over get_net_out (PyTorch): {get_net_out_original_time_ms / get_net_out_triton_l2_time_ms:.2f}x")

        match_triton_l2_ref = torch.allclose(out_net_original, out_net_triton_l2, atol=1e-3, rtol=1e-3)
        rel_error_l2 = torch.norm(out_net_original - out_net_triton_l2) / (torch.norm(out_net_original) + 1e-8)
        print(
            "[Sanity Check] Do Triton L2 and PyTorch provide the identical net_out tensor? "
            f"-> {'YES' if match_triton_l2_ref else 'NO!!!'}"
        )
        print(f"Relative error between PyTorch and Triton L2 outputs: {rel_error_l2:.2e}")
        
        if not match_triton_l2_ref:
            diff = out_net_original - out_net_triton_l2
            max_diff = torch.max(torch.abs(diff))
            indices = torch.where(torch.abs(diff) == max_diff)
            e_idx = indices[0][0].item()
            c_idx = indices[1][0].item()
            print(f"  MAX DIFF: {max_diff:.4f} at edge {e_idx}, col {c_idx}")
            print(f"  Orig val: {out_net_original[e_idx, c_idx]:.4f}")
            print(f"  Triton val: {out_net_triton_l2[e_idx, c_idx]:.4f}")
            
        net_out_results.append({
            "name": "Triton naive L2 get_net_out",
            "basis": basis_name,
            "time_ms": get_net_out_triton_l2_time_ms,
            "speedup_vs_original": get_net_out_original_time_ms / get_net_out_triton_l2_time_ms,
            "match": match_triton_l2_ref
        })
        
    try:
        _print_autotune_best(
            foo2,
            "triton_tensor_decomp_l2_final",
            ["get_net_out_triton_block_kernel_l2"],
        )
    except Exception:
        pass
    
    return impl_results + grouped_results + net_out_results


def run_benchmark():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32

    print(f"Running on device: {device}")
    
    # Collect all basis sets to benchmark
    basis_sets_to_test = {
        "orbital_basis_def2_svp_nabla": basis_sets_module.orbital_basis_def2_svp_nabla,
        "orbital_basis_def2_svp_QM7": basis_sets_module.orbital_basis_def2_svp_QM7,
        "def2_tzvpd (subset)": {
            1: basis_sets_module.def2_tzvpd.get('H', []),
            6: basis_sets_module.def2_tzvpd.get('C', []),
            8: basis_sets_module.def2_tzvpd.get('O', []),
        },
    }
    
    all_results = []
    
    num_edges_list = [10000, 50000, 100000, 200000]
    
    for num_edges in num_edges_list:
        print(f"\n\n{'#'*80}")
        print(f"RUNNING BENCHMARKS FOR NUM_EDGES = {num_edges}")
        print(f"{'#'*80}")
        
        for basis_name, orbital_basis in basis_sets_to_test.items():
            try:
                results = run_benchmark_for_basis(basis_name, orbital_basis, device, dtype, num_edges=num_edges)
                # Add num_edges to the results for the summary
                for r in results:
                    r["num_edges"] = num_edges
                all_results.extend(results)
            except Exception as e:
                print(f"\nERROR benchmarking {basis_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Print final cross-basis summary
    print(f"\n\n{'='*100}")
    print("FINAL CROSS-BASIS SUMMARY")
    print(f"{'='*100}\n")
    cross_headers = ["NUM EDGES", "BASIS SET", "IMPLEMENTATION", "TIME (ms)", "TFLOPS", "SPEEDUP", "MATCH"]
    cross_rows = []
    for r in all_results:
        impl_name = r.get("name", "")
        basis = r.get("basis", "")
        edges = r.get("num_edges", 0)
        time_ms = r.get("time_ms", 0)
        tflops = r.get("tflops", 0)
        speedup = r.get("speedup_vs_original", r.get("vs_original", 1.0))
        match = r.get("match", False)
        cross_rows.append([
            str(edges),
            basis,
            impl_name,
            f"{time_ms:.3f}",
            f"{tflops:.3f}",
            f"{speedup:.2f}",
            str(match),
        ])
    _print_aligned_table(
        cross_headers,
        cross_rows,
        right_align=[False, False, False, True, True, True, True],
    )


if __name__ == "__main__":
    run_benchmark()
