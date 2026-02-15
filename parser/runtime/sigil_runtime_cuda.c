/*
 * Sigil CUDA Runtime Library
 *
 * Provides GPU compute functions for AOT-compiled Sigil programs.
 * Uses CUDA Driver API for maximum control and compatibility.
 *
 * Build: nvcc -c sigil_runtime_cuda.c -o sigil_runtime_cuda.o
 * Link with: -lcuda -lnvrtc
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cuda.h>
#include <nvrtc.h>

/* Global CUDA state */
static CUcontext g_cuda_context = NULL;
static CUdevice g_cuda_device = 0;
static int g_cuda_initialized = 0;

/* Error checking macros */
#define CUDA_CHECK(call) \
    do { \
        CUresult err = (call); \
        if (err != CUDA_SUCCESS) { \
            const char* errStr; \
            cuGetErrorString(err, &errStr); \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", errStr, __FILE__, __LINE__); \
            return -1; \
        } \
    } while(0)

#define NVRTC_CHECK(call) \
    do { \
        nvrtcResult err = (call); \
        if (err != NVRTC_SUCCESS) { \
            fprintf(stderr, "NVRTC error: %s at %s:%d\n", nvrtcGetErrorString(err), __FILE__, __LINE__); \
            return -1; \
        } \
    } while(0)

/* ============================================================================
 * Initialization and Cleanup
 * ============================================================================ */

/* Initialize CUDA - returns 1 on success, 0 on failure */
int64_t sigil_cuda_init(void) {
    if (g_cuda_initialized) return 1;

    CUresult err = cuInit(0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to initialize CUDA\n");
        return 0;
    }

    int deviceCount;
    err = cuDeviceGetCount(&deviceCount);
    if (err != CUDA_SUCCESS || deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return 0;
    }

    err = cuDeviceGet(&g_cuda_device, 0);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to get CUDA device\n");
        return 0;
    }

    err = cuCtxCreate(&g_cuda_context, 0, g_cuda_device);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to create CUDA context\n");
        return 0;
    }

    g_cuda_initialized = 1;
    return 1;
}

/* Cleanup CUDA resources */
void sigil_cuda_cleanup(void) {
    if (g_cuda_context) {
        cuCtxDestroy(g_cuda_context);
        g_cuda_context = NULL;
    }
    g_cuda_initialized = 0;
}

/* Get number of CUDA devices */
int64_t sigil_cuda_get_device_count(void) {
    int count = 0;
    if (cuDeviceGetCount(&count) != CUDA_SUCCESS) {
        return 0;
    }
    return (int64_t)count;
}

/* ============================================================================
 * Memory Management
 * ============================================================================ */

/* Allocate device memory - returns device pointer or 0 on failure */
int64_t sigil_cuda_malloc(int64_t size) {
    if (!g_cuda_initialized) {
        if (!sigil_cuda_init()) return 0;
    }

    CUdeviceptr dptr;
    if (cuMemAlloc(&dptr, (size_t)size) != CUDA_SUCCESS) {
        return 0;
    }
    return (int64_t)dptr;
}

/* Free device memory */
void sigil_cuda_free(int64_t device_ptr) {
    if (device_ptr != 0) {
        cuMemFree((CUdeviceptr)device_ptr);
    }
}

/* Copy host to device - returns 0 on success, -1 on failure */
int64_t sigil_cuda_memcpy_h2d(int64_t dst, void* src, int64_t size) {
    if (!g_cuda_initialized) return -1;
    CUDA_CHECK(cuMemcpyHtoD((CUdeviceptr)dst, src, (size_t)size));
    return 0;
}

/* Copy device to host - returns 0 on success, -1 on failure */
int64_t sigil_cuda_memcpy_d2h(void* dst, int64_t src, int64_t size) {
    if (!g_cuda_initialized) return -1;
    CUDA_CHECK(cuMemcpyDtoH(dst, (CUdeviceptr)src, (size_t)size));
    return 0;
}

/* Copy device to device - returns 0 on success, -1 on failure */
int64_t sigil_cuda_memcpy_d2d(int64_t dst, int64_t src, int64_t size) {
    if (!g_cuda_initialized) return -1;
    CUDA_CHECK(cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, (size_t)size));
    return 0;
}

/* Synchronize - wait for all GPU operations to complete */
void sigil_cuda_sync(void) {
    if (g_cuda_initialized) {
        cuCtxSynchronize();
    }
}

/* ============================================================================
 * Kernel Compilation and Execution
 * ============================================================================ */

/* Compile CUDA source to PTX and load kernel - returns handle or -1 on failure */
int64_t sigil_cuda_compile_kernel(const char* cuda_src, const char* kernel_name) {
    if (!g_cuda_initialized) {
        if (!sigil_cuda_init()) return -1;
    }

    /* Create NVRTC program */
    nvrtcProgram prog;
    nvrtcResult nvrtc_err = nvrtcCreateProgram(&prog, cuda_src, "kernel.cu", 0, NULL, NULL);
    if (nvrtc_err != NVRTC_SUCCESS) {
        fprintf(stderr, "Failed to create NVRTC program\n");
        return -1;
    }

    /* Compile to PTX */
    const char* opts[] = {"--gpu-architecture=compute_70"};  /* Volta and newer */
    nvrtc_err = nvrtcCompileProgram(prog, 1, opts);
    if (nvrtc_err != NVRTC_SUCCESS) {
        /* Get compilation log */
        size_t logSize;
        nvrtcGetProgramLogSize(prog, &logSize);
        char* log = (char*)malloc(logSize);
        nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "NVRTC compilation failed:\n%s\n", log);
        free(log);
        nvrtcDestroyProgram(&prog);
        return -1;
    }

    /* Get PTX */
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = (char*)malloc(ptxSize);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    /* Load module from PTX */
    CUmodule module;
    CUresult err = cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
    free(ptx);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to load CUDA module from PTX\n");
        return -1;
    }

    /* Get kernel function */
    CUfunction kernel;
    err = cuModuleGetFunction(&kernel, module, kernel_name);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to get kernel function '%s'\n", kernel_name);
        cuModuleUnload(module);
        return -1;
    }

    /* Return kernel handle (we keep module loaded) */
    return (int64_t)kernel;
}

/* Load pre-compiled PTX and get kernel - returns handle or -1 on failure */
int64_t sigil_cuda_load_ptx(const char* ptx, const char* kernel_name) {
    if (!g_cuda_initialized) {
        if (!sigil_cuda_init()) return -1;
    }

    CUmodule module;
    CUresult err = cuModuleLoadDataEx(&module, ptx, 0, NULL, NULL);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to load PTX module\n");
        return -1;
    }

    CUfunction kernel;
    err = cuModuleGetFunction(&kernel, module, kernel_name);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "Failed to get kernel function '%s'\n", kernel_name);
        cuModuleUnload(module);
        return -1;
    }

    return (int64_t)kernel;
}

/* Launch 1D kernel - returns 0 on success, -1 on failure
 * args is an array of void* pointers to kernel arguments
 */
int64_t sigil_cuda_launch_kernel_1d(int64_t kernel_handle, int64_t grid_x,
                                     int64_t block_x, void** args, int64_t num_args) {
    if (!g_cuda_initialized) return -1;

    CUfunction kernel = (CUfunction)kernel_handle;

    CUresult err = cuLaunchKernel(
        kernel,
        (unsigned int)grid_x, 1, 1,    /* Grid dimensions */
        (unsigned int)block_x, 1, 1,   /* Block dimensions */
        0,                              /* Shared memory */
        NULL,                           /* Stream (default) */
        args,                           /* Kernel arguments */
        NULL                            /* Extra */
    );

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "Kernel launch failed: %s\n", errStr);
        return -1;
    }

    return 0;
}

/* Launch 2D kernel - returns 0 on success, -1 on failure */
int64_t sigil_cuda_launch_kernel_2d(int64_t kernel_handle,
                                     int64_t grid_x, int64_t grid_y,
                                     int64_t block_x, int64_t block_y,
                                     void** args, int64_t num_args) {
    if (!g_cuda_initialized) return -1;

    CUfunction kernel = (CUfunction)kernel_handle;

    CUresult err = cuLaunchKernel(
        kernel,
        (unsigned int)grid_x, (unsigned int)grid_y, 1,
        (unsigned int)block_x, (unsigned int)block_y, 1,
        0, NULL, args, NULL
    );

    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "Kernel launch failed: %s\n", errStr);
        return -1;
    }

    return 0;
}

/* ============================================================================
 * Utility Functions
 * ============================================================================ */

/* Get device name - returns pointer to static buffer */
const char* sigil_cuda_get_device_name(void) {
    static char name[256] = {0};
    if (g_cuda_initialized || sigil_cuda_init()) {
        cuDeviceGetName(name, sizeof(name), g_cuda_device);
    }
    return name;
}

/* Get device compute capability - returns major * 10 + minor */
int64_t sigil_cuda_get_compute_capability(void) {
    if (!g_cuda_initialized && !sigil_cuda_init()) return 0;

    int major, minor;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, g_cuda_device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, g_cuda_device);
    return major * 10 + minor;
}

/* Get total device memory in bytes */
int64_t sigil_cuda_get_total_memory(void) {
    if (!g_cuda_initialized && !sigil_cuda_init()) return 0;

    size_t total;
    cuDeviceTotalMem(&total, g_cuda_device);
    return (int64_t)total;
}

/* ============================================================================
 * Nihil-compatible aliases (cuda_* instead of sigil_cuda_*)
 * These match the extern "C" declarations in nihil-cuda
 * ============================================================================ */

void cuda_init(size_t device) {
    (void)device;  /* Currently single-device */
    sigil_cuda_init();
}

uint8_t* cuda_malloc(size_t size) {
    return (uint8_t*)sigil_cuda_malloc((int64_t)size);
}

void cuda_free(uint8_t* ptr) {
    sigil_cuda_free((int64_t)(uintptr_t)ptr);
}

void cuda_memset(uint8_t* ptr, int32_t value, size_t size) {
    if (!g_cuda_initialized) return;
    cuMemsetD8((CUdeviceptr)ptr, (unsigned char)value, size);
}

void cuda_memcpy_htod(uint8_t* dst, const uint8_t* src, size_t size) {
    sigil_cuda_memcpy_h2d((int64_t)(uintptr_t)dst, (void*)src, (int64_t)size);
}

void cuda_memcpy_dtoh(uint8_t* dst, const uint8_t* src, size_t size) {
    sigil_cuda_memcpy_d2h(dst, (int64_t)(uintptr_t)src, (int64_t)size);
}

void cuda_memcpy_dtod(uint8_t* dst, const uint8_t* src, size_t size) {
    sigil_cuda_memcpy_d2d((int64_t)(uintptr_t)dst, (int64_t)(uintptr_t)src, (int64_t)size);
}

void cuda_device_synchronize(void) {
    sigil_cuda_sync();
}

/* Returns (free, total) memory - packed as two size_t values */
typedef struct { size_t free; size_t total; } CudaMemInfo;
CudaMemInfo cuda_mem_get_info(void) {
    CudaMemInfo info = {0, 0};
    if (g_cuda_initialized) {
        cuMemGetInfo(&info.free, &info.total);
    }
    return info;
}

/* Device properties structure matching Nihil's CudaDeviceProperties */
typedef struct {
    char name[256];
    size_t total_memory;
    int32_t compute_major;
    int32_t compute_minor;
    int32_t multiprocessor_count;
    int32_t max_threads_per_block;
    int32_t warp_size;
} CudaDeviceProperties;

CudaDeviceProperties cuda_get_device_properties(size_t device) {
    CudaDeviceProperties props = {0};
    if (!g_cuda_initialized && !sigil_cuda_init()) return props;

    CUdevice dev;
    if (cuDeviceGet(&dev, (int)device) != CUDA_SUCCESS) return props;

    cuDeviceGetName(props.name, sizeof(props.name), dev);
    cuDeviceTotalMem(&props.total_memory, dev);
    cuDeviceGetAttribute(&props.compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    cuDeviceGetAttribute(&props.compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    cuDeviceGetAttribute(&props.multiprocessor_count, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
    cuDeviceGetAttribute(&props.max_threads_per_block, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, dev);
    cuDeviceGetAttribute(&props.warp_size, CU_DEVICE_ATTRIBUTE_WARP_SIZE, dev);

    return props;
}

/* Forward declarations for functions from sigil_runtime.c */
extern void* sigil_vec_new(int64_t capacity);
extern void sigil_vec_push(void* vec_ptr, int64_t value);

/* Compile PTX string to cubin - returns Vec<u8> (as SigilVec*) */
void* cuda_compile_ptx(const uint8_t* ptx, size_t len) {
    if (!g_cuda_initialized && !sigil_cuda_init()) return NULL;

    /* For now, just load the PTX directly - cuModuleLoadDataEx handles PTX */
    /* In a full implementation, we'd use nvJitLink or similar */

    /* Create a Vec to hold the PTX (which can be loaded as-is) */
    /* This is a simplified version - real impl would compile to cubin */
    void* vec = sigil_vec_new((int64_t)len);
    if (!vec) return NULL;

    for (size_t i = 0; i < len; i++) {
        sigil_vec_push(vec, (int64_t)ptx[i]);
    }
    return vec;
}

/* CudaModule structure */
typedef struct {
    uint64_t handle;  /* CUmodule */
} CudaModule;

/* Load cubin/PTX and return module handle */
CudaModule cuda_load_module(const uint8_t* data, size_t len) {
    CudaModule mod = {0};
    if (!g_cuda_initialized && !sigil_cuda_init()) return mod;

    CUmodule module;
    CUresult err = cuModuleLoadDataEx(&module, data, 0, NULL, NULL);
    if (err == CUDA_SUCCESS) {
        mod.handle = (uint64_t)module;
    }
    return mod;
}

/* ============================================================================
 * Kernel Launch Stubs (for Nihil integration)
 * These are placeholder implementations - real kernels would be CUDA code
 * ============================================================================ */

/* GEMM kernel stubs */
void launch_gemm_fp16_kernel(
    const void* a, const void* b, void* c, const void* d,
    float alpha, float beta,
    int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    const void* config
) {
    fprintf(stderr, "STUB: launch_gemm_fp16_kernel called (M=%lld, N=%lld, K=%lld)\n",
            (long long)m, (long long)n, (long long)k);
    /* TODO: Implement actual CUDA kernel */
}

void launch_gemm_fp8_kernel(
    const void* a, const void* b, void* c, const void* d,
    float alpha, float beta,
    int64_t m, int64_t n, int64_t k,
    int64_t lda, int64_t ldb, int64_t ldc,
    const void* config
) {
    fprintf(stderr, "STUB: launch_gemm_fp8_kernel called\n");
}

void launch_batch_gemm_fp16_kernel(
    const void* a, const void* b, void* c, const void* d,
    float alpha, float beta,
    int64_t m, int64_t n, int64_t k,
    int64_t batch, int64_t stride_a, int64_t stride_b, int64_t stride_c,
    const void* config
) {
    fprintf(stderr, "STUB: launch_batch_gemm_fp16_kernel called\n");
}

void launch_gemm_fused_kernel(
    const void* a, const void* b, const void* bias, const void* residual,
    void* output,
    int64_t m, int64_t n, int64_t k,
    int64_t activation, float alpha, float beta,
    const void* config
) {
    fprintf(stderr, "STUB: launch_gemm_fused_kernel called\n");
}

/* Flash Attention kernel stubs */
void launch_flash_attn_fwd_kernel(
    const void* q, const void* k, const void* v, void* out,
    int64_t batch, int64_t heads, int64_t seq_len, int64_t head_dim,
    float scale, int64_t causal,
    const void* config
) {
    fprintf(stderr, "STUB: launch_flash_attn_fwd_kernel called (B=%lld, H=%lld, S=%lld, D=%lld)\n",
            (long long)batch, (long long)heads, (long long)seq_len, (long long)head_dim);
}

void launch_flash_attn_bwd_kernel(
    const void* dout, const void* q, const void* k, const void* v,
    const void* out, const void* softmax_lse,
    void* dq, void* dk, void* dv,
    int64_t batch, int64_t heads, int64_t seq_len, int64_t head_dim,
    float scale, int64_t causal,
    const void* config
) {
    fprintf(stderr, "STUB: launch_flash_attn_bwd_kernel called\n");
}

void launch_flash_attn_varlen_kernel(
    const void* q, const void* k, const void* v, void* out,
    const void* cu_seqlens_q, const void* cu_seqlens_k,
    int64_t max_seqlen_q, int64_t max_seqlen_k,
    int64_t batch, int64_t heads, int64_t head_dim,
    float scale, int64_t causal,
    const void* config
) {
    fprintf(stderr, "STUB: launch_flash_attn_varlen_kernel called\n");
}

void launch_flash_attn_gqa_kernel(
    const void* q, const void* k, const void* v, void* out,
    int64_t batch, int64_t q_heads, int64_t kv_heads,
    int64_t seq_len, int64_t head_dim,
    float scale, int64_t causal,
    const void* config
) {
    fprintf(stderr, "STUB: launch_flash_attn_gqa_kernel called\n");
}

void launch_paged_attention_kernel(
    const void* q, const void* k_cache, const void* v_cache,
    const void* block_tables, const void* seq_lens,
    void* out,
    int64_t batch, int64_t heads, int64_t head_dim,
    int64_t block_size, int64_t max_blocks,
    float scale,
    const void* config
) {
    fprintf(stderr, "STUB: launch_paged_attention_kernel called\n");
}

/* FP8 kernel stubs */
void launch_fp8_e4m3_gemm_kernel(
    const void* a, const void* b, void* c,
    int64_t m, int64_t n, int64_t k,
    float scale_a, float scale_b
) {
    fprintf(stderr, "STUB: launch_fp8_e4m3_gemm_kernel called\n");
}

void launch_fp8_e5m2_gemm_kernel(
    const void* a, const void* b, void* c,
    int64_t m, int64_t n, int64_t k,
    float scale_a, float scale_b
) {
    fprintf(stderr, "STUB: launch_fp8_e5m2_gemm_kernel called\n");
}

void launch_quantize_e4m3_kernel(
    const void* input, void* output, void* scale,
    int64_t numel
) {
    fprintf(stderr, "STUB: launch_quantize_e4m3_kernel called\n");
}

void launch_quantize_e5m2_kernel(
    const void* input, void* output, void* scale,
    int64_t numel
) {
    fprintf(stderr, "STUB: launch_quantize_e5m2_kernel called\n");
}
