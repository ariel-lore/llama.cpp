#pragma once

#include <cstddef>

// Opt-in (--cuda-d3d12-resident / GGML_CUDA_D3D12_RESIDENT), Windows only:
// allocate a device buffer as a shared D3D12 heap with maximum residency priority
// and import it into CUDA. On other platforms or on any failure
// ggml_cuda_d3d12_resident_alloc returns false and the caller uses cudaMalloc.
void ggml_cuda_d3d12_resident_set(bool enabled);
bool ggml_cuda_d3d12_resident_enabled();
bool ggml_cuda_d3d12_resident_alloc(int device, size_t size, void ** dev_ptr, void ** handle);
void ggml_cuda_d3d12_resident_free(void * handle);
