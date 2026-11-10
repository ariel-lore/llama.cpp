#include "d3d12-resident.cuh"

#include "ggml-impl.h"

#include <cstdlib>
#include <cstring>

// Set by ggml_backend_cuda_set_d3d12_resident (CLI / config.ini) before model load.
// If the setter was never called, fall back to the env var for library users.
static bool g_d3d12_resident_set     = false;
static bool g_d3d12_resident_enabled = false;

void ggml_cuda_d3d12_resident_set(bool enabled) {
    g_d3d12_resident_set     = true;
    g_d3d12_resident_enabled = enabled;
}

bool ggml_cuda_d3d12_resident_enabled() {
#ifdef _WIN32
    if (g_d3d12_resident_set) {
        return g_d3d12_resident_enabled;
    }
    const char * v = getenv("GGML_CUDA_D3D12_RESIDENT");
    return v != nullptr && strcmp(v, "0") != 0;
#else
    return false;
#endif
}

#ifndef _WIN32

bool ggml_cuda_d3d12_resident_alloc(int, size_t, void **, void **) { return false; }
void ggml_cuda_d3d12_resident_free(void *) {}

#else

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <cuda_runtime.h>

#include <mutex>

namespace {

struct d3d12_alloc {
    ID3D12Heap *         heap    = nullptr;
    cudaExternalMemory_t ext_mem = nullptr;
    void *               dev_ptr = nullptr;
};

typedef HRESULT (WINAPI * pfn_create_dxgi_factory1)(REFIID, void **);

constexpr int MAX_DEVICES = 16;
std::mutex    g_mutex;
ID3D12Device * g_devices[MAX_DEVICES] = {};
bool           g_tried[MAX_DEVICES]   = {};

// one D3D12 device per CUDA device, matched by adapter LUID; d3d12.dll/dxgi.dll are loaded at runtime
ID3D12Device * get_device(int device) {
    if (device < 0 || device >= MAX_DEVICES) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_tried[device]) {
        return g_devices[device];
    }
    g_tried[device] = true;

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        GGML_LOG_WARN("%s: cudaGetDeviceProperties failed\n", __func__);
        return nullptr;
    }
    LUID luid;
    static_assert(sizeof(luid) == sizeof(prop.luid), "LUID size mismatch");
    memcpy(&luid, prop.luid, sizeof(luid));

    HMODULE dxgi  = LoadLibraryA("dxgi.dll");
    HMODULE d3d12 = LoadLibraryA("d3d12.dll");
    if (!dxgi || !d3d12) {
        GGML_LOG_WARN("%s: failed to load dxgi.dll/d3d12.dll\n", __func__);
        return nullptr;
    }
    auto create_factory = (pfn_create_dxgi_factory1) GetProcAddress(dxgi, "CreateDXGIFactory1");
    auto create_device  = (PFN_D3D12_CREATE_DEVICE)  GetProcAddress(d3d12, "D3D12CreateDevice");
    if (!create_factory || !create_device) {
        GGML_LOG_WARN("%s: missing CreateDXGIFactory1/D3D12CreateDevice\n", __func__);
        return nullptr;
    }

    IDXGIFactory4 * factory = nullptr;
    HRESULT hr = create_factory(__uuidof(IDXGIFactory4), (void **) &factory);
    if (FAILED(hr)) {
        GGML_LOG_WARN("%s: CreateDXGIFactory1 failed: 0x%08lx\n", __func__, (unsigned long) hr);
        return nullptr;
    }
    IDXGIAdapter1 * adapter = nullptr;
    hr = factory->EnumAdapterByLuid(luid, __uuidof(IDXGIAdapter1), (void **) &adapter);
    factory->Release();
    if (FAILED(hr)) {
        GGML_LOG_WARN("%s: no DXGI adapter matches CUDA device %d LUID: 0x%08lx\n", __func__, device, (unsigned long) hr);
        return nullptr;
    }
    ID3D12Device * dev = nullptr;
    hr = create_device(adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), (void **) &dev);
    adapter->Release();
    if (FAILED(hr)) {
        GGML_LOG_WARN("%s: D3D12CreateDevice failed: 0x%08lx\n", __func__, (unsigned long) hr);
        return nullptr;
    }
    g_devices[device] = dev;
    return dev;
}

void release(d3d12_alloc * a) {
    if (a->dev_ptr) {
        cudaFree(a->dev_ptr);
    }
    if (a->ext_mem) {
        cudaDestroyExternalMemory(a->ext_mem);
    }
    if (a->heap) {
        a->heap->Release();
    }
    delete a;
}

} // namespace

bool ggml_cuda_d3d12_resident_alloc(int device, size_t size, void ** dev_ptr, void ** handle) {
    ID3D12Device * dev = get_device(device);
    if (!dev) {
        return false;
    }

    const UINT64 align   = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT; // 64 KiB
    const UINT64 aligned = ((UINT64) size + align - 1) / align * align;

    D3D12_HEAP_DESC desc = {};
    desc.SizeInBytes     = aligned;
    desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    desc.Alignment       = align;
    desc.Flags           = D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;

    d3d12_alloc * a = new d3d12_alloc();
    HRESULT hr = dev->CreateHeap(&desc, __uuidof(ID3D12Heap), (void **) &a->heap);
    if (FAILED(hr)) {
        GGML_LOG_WARN("%s: CreateHeap(%.2f MiB) failed: 0x%08lx\n", __func__, aligned / 1048576.0, (unsigned long) hr);
        release(a);
        return false;
    }

    const char * prio_name = "none";
    ID3D12Device1 * dev1 = nullptr;
    if (SUCCEEDED(dev->QueryInterface(__uuidof(ID3D12Device1), (void **) &dev1))) {
        ID3D12Pageable * obj = a->heap;
        D3D12_RESIDENCY_PRIORITY prio = D3D12_RESIDENCY_PRIORITY_MAXIMUM;
        if (SUCCEEDED(dev1->SetResidencyPriority(1, &obj, &prio))) {
            prio_name = "MAXIMUM";
        } else {
            prio = D3D12_RESIDENCY_PRIORITY_HIGH; // MAXIMUM is documented as not guaranteed to be available
            if (SUCCEEDED(dev1->SetResidencyPriority(1, &obj, &prio))) {
                prio_name = "HIGH";
            }
        }
        dev1->Release();
    }

    HANDLE shared = nullptr;
    hr = dev->CreateSharedHandle(a->heap, nullptr, GENERIC_ALL, nullptr, &shared);
    if (FAILED(hr)) {
        GGML_LOG_WARN("%s: CreateSharedHandle failed: 0x%08lx\n", __func__, (unsigned long) hr);
        release(a);
        return false;
    }

    cudaExternalMemoryHandleDesc hdesc = {};
    hdesc.type                = cudaExternalMemoryHandleTypeD3D12Heap;
    hdesc.handle.win32.handle = shared;
    hdesc.size                = aligned;
    cudaError_t err = cudaImportExternalMemory(&a->ext_mem, &hdesc);
    CloseHandle(shared);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: cudaImportExternalMemory failed: %s\n", __func__, cudaGetErrorString(err));
        a->ext_mem = nullptr;
        release(a);
        return false;
    }

    cudaExternalMemoryBufferDesc bdesc = {};
    bdesc.offset = 0;
    bdesc.size   = aligned;
    err = cudaExternalMemoryGetMappedBuffer(&a->dev_ptr, a->ext_mem, &bdesc);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_WARN("%s: cudaExternalMemoryGetMappedBuffer failed: %s\n", __func__, cudaGetErrorString(err));
        a->dev_ptr = nullptr;
        release(a);
        return false;
    }

    GGML_LOG_INFO("%s: device %d: D3D12 heap %.2f MiB, residency priority %s, imported into CUDA\n",
                  __func__, device, aligned / 1048576.0, prio_name);
    *dev_ptr = a->dev_ptr;
    *handle  = a;
    return true;
}

void ggml_cuda_d3d12_resident_free(void * handle) {
    if (handle) {
        release((d3d12_alloc *) handle);
    }
}

#endif // _WIN32
