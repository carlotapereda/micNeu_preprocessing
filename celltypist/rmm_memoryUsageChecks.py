import rmm
import cupy as cp

from rmm.allocators.cupy import rmm_cupy_allocator

# -------------------------------------------
# OPTION 1 — Unified Virtual Memory (UVM)
# -------------------------------------------
# Enables paging to host RAM if GPU VRAM runs out (slower but safer).
# Ideal when your dataset may exceed GPU memory.
# allows to process datasets larger than GPU VRAM
rmm.reinitialize(
    managed_memory=True,   # enables Unified Virtual Memory
    pool_allocator=False,  # optional (pooling not needed with UVM)
    devices=0
)
cp.cuda.set_allocator(rmm_cupy_allocator)

# To confirm:
print(">>> Using UVM allocator")
print(rmm.mr.get_current_device_resource())
print(rmm.get_info())

# -------------------------------------------
# OPTION 2 — GPU Memory Pool (fast, strict)
# -------------------------------------------
# ⚠️ Only use ONE initialization per session.
# If you want to test this instead, restart the Python process first.
# This is fastest, but will raise CUDA OOM if you exceed GPU VRAM.

# --- RAPIDS memory pool ---
# This has no CPU fallback
# This is fastest but can crash due to out-of-memory errors
rmm.reinitialize(
    managed_memory=False,
    pool_allocator=True,
    devices=0,
)
cp.cuda.set_allocator(rmm_cupy_allocator)

#After initializing RMM (with or without managed memory), you can use:
info = rmm.mr.get_current_device_resource()
print(info)
#This shows what memory resource (allocator) RMM is using — e.g., whether it’s PoolMemoryResource or ManagedMemoryResource.


rmm.get_info() #RAPIDS exposes a convenient summary
# Total = total GPU memory in bytes
# Free =  free memory
# Used = currently allocated




# -------------------------------------------
# CuPy memory pool diagnostics
# -------------------------------------------
#CuPy (which uses RMM as its allocator) can also report memory stats:

mempool = cp.get_default_memory_pool()
print(f"CuPy used memory : {mempool.used_bytes() / 1e9:.2f} GB")
print(f"CuPy total pool  : {mempool.total_bytes() / 1e9:.2f} GB")

# Optional: free all unused GPU allocations
mempool.free_all_blocks()

