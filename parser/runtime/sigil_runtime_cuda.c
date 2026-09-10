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
#include <math.h>
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

/* Fill device buffer with N(0,1) random values via host staging.
 * Avoids passing &StoragePtr (fat-pointer ABI bug): takes raw device_ptr + n. */
void sigil_cuda_fill_randn_f32(int64_t device_ptr, int64_t n) {
    if (n <= 0 || !device_ptr) return;
    float* host = (float*)malloc((size_t)n * sizeof(float));
    if (!host) return;
    for (int64_t i = 0; i + 1 < n; i += 2) {
        double u1, u2;
        do { u1 = (double)rand() / ((double)RAND_MAX + 1.0); } while (u1 < 1e-10);
        u2 = (double)rand() / ((double)RAND_MAX + 1.0);
        double r = sqrt(-2.0 * log(u1));
        double t = 6.28318530718 * u2;
        host[i]     = (float)(r * cos(t));
        host[i + 1] = (float)(r * sin(t));
    }
    if (n & 1) { host[n-1] = host[0]; }
    cuMemcpyHtoD((CUdeviceptr)device_ptr, host, (size_t)n * sizeof(float));
    free(host);
}

/* Fill device buffer with zeros. */
void sigil_cuda_zero_f32(int64_t device_ptr, int64_t n) {
    if (n <= 0 || !device_ptr) return;
    cuMemsetD8((CUdeviceptr)device_ptr, 0, (size_t)n * sizeof(float));
}

/* Zero exactly `bytes` bytes of device memory (byte-granularity, for non-float dtypes). */
void sigil_cuda_memset_zero(int64_t device_ptr, int64_t bytes) {
    if (bytes <= 0 || !device_ptr) return;
    cuMemsetD8((CUdeviceptr)device_ptr, 0, (size_t)bytes);
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

/* ============================================================================
 * Essential Tensor Kernel Launchers (for nihil-cuda Tensor operations)
 * ============================================================================ */

/* KernelConfig structure matching nihil-cuda */
typedef struct {
    int64_t grid_x;
    int64_t grid_y;
    int64_t grid_z;
    int64_t block_x;
    int64_t block_y;
    int64_t block_z;
    int64_t shared_mem;
} KernelConfig;

/* XorShift64 PRNG state */
static uint64_t g_randn_state = 0x853c49e6748fea9bULL;

/* Box-Muller transform for generating normal distribution */
static void generate_normal_pair(float* z0, float* z1) {
    static const double TWO_PI = 6.283185307179586;

    /* Generate two uniform random numbers in (0, 1] */
    g_randn_state ^= g_randn_state >> 12;
    g_randn_state ^= g_randn_state << 25;
    g_randn_state ^= g_randn_state >> 27;
    double u1 = (double)(g_randn_state * 0x2545F4914F6CDD1DULL) / (double)UINT64_MAX;

    g_randn_state ^= g_randn_state >> 12;
    g_randn_state ^= g_randn_state << 25;
    g_randn_state ^= g_randn_state >> 27;
    double u2 = (double)(g_randn_state * 0x2545F4914F6CDD1DULL) / (double)UINT64_MAX;

    /* Avoid log(0) */
    if (u1 < 1e-10) u1 = 1e-10;

    /* Box-Muller transform */
    double mag = sqrt(-2.0 * log(u1));
    *z0 = (float)(mag * cos(TWO_PI * u2));
    *z1 = (float)(mag * sin(TWO_PI * u2));
}

/* Fill tensor with random normal values (mean=0, std=1) */
void launch_randn_kernel(void* ptr, int64_t numel) {
    if (!g_cuda_initialized && !sigil_cuda_init()) {
        fprintf(stderr, "launch_randn_kernel: CUDA not initialized\n");
        return;
    }

    /* Allocate host buffer */
    float* host_data = (float*)malloc(numel * sizeof(float));
    if (!host_data) {
        fprintf(stderr, "launch_randn_kernel: malloc failed\n");
        return;
    }

    /* Generate random normal values on host */
    int64_t i;
    for (i = 0; i < numel - 1; i += 2) {
        generate_normal_pair(&host_data[i], &host_data[i + 1]);
    }
    /* Handle odd count */
    if (numel % 2 == 1) {
        float z0, z1;
        generate_normal_pair(&z0, &z1);
        host_data[numel - 1] = z0;
    }

    /* Copy to device */
    CUresult err = cuMemcpyHtoD((CUdeviceptr)ptr, host_data, numel * sizeof(float));
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "launch_randn_kernel: cuMemcpyHtoD failed: %s\n", errStr);
    }

    free(host_data);
}

/* Fill tensor with a constant value */
void launch_fill_kernel(void* ptr, float value, int64_t numel, KernelConfig config) {
    (void)config;  /* Not used for this simple impl */

    if (!g_cuda_initialized && !sigil_cuda_init()) {
        fprintf(stderr, "launch_fill_kernel: CUDA not initialized\n");
        return;
    }

    /* Use cuMemsetD32 for float fill (reinterpret float as uint32) */
    uint32_t value_bits;
    memcpy(&value_bits, &value, sizeof(value_bits));

    CUresult err = cuMemsetD32((CUdeviceptr)ptr, value_bits, numel);
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "launch_fill_kernel: cuMemsetD32 failed: %s\n", errStr);
    }
}

/* Fill tensor with uniform random values in [low, high) */
void launch_uniform_kernel(void* ptr, float low, float high, int64_t numel) {
    if (!g_cuda_initialized && !sigil_cuda_init()) {
        fprintf(stderr, "launch_uniform_kernel: CUDA not initialized\n");
        return;
    }

    float range = high - low;

    /* Allocate host buffer */
    float* host_data = (float*)malloc(numel * sizeof(float));
    if (!host_data) {
        fprintf(stderr, "launch_uniform_kernel: malloc failed\n");
        return;
    }

    /* Generate uniform values on host */
    for (int64_t i = 0; i < numel; i++) {
        g_randn_state ^= g_randn_state >> 12;
        g_randn_state ^= g_randn_state << 25;
        g_randn_state ^= g_randn_state >> 27;
        double u = (double)(g_randn_state * 0x2545F4914F6CDD1DULL) / (double)UINT64_MAX;
        host_data[i] = low + (float)(u * range);
    }

    /* Copy to device */
    CUresult err = cuMemcpyHtoD((CUdeviceptr)ptr, host_data, numel * sizeof(float));
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "launch_uniform_kernel: cuMemcpyHtoD failed: %s\n", errStr);
    }

    free(host_data);
}

/* Cast tensor from one dtype to another */
void launch_cast_kernel(const void* src, void* dst, int64_t numel) {
    if (!g_cuda_initialized && !sigil_cuda_init()) {
        fprintf(stderr, "launch_cast_kernel: CUDA not initialized\n");
        return;
    }

    /* For now, just do a memcpy - proper implementation would handle type conversion */
    /* This assumes src and dst have same element size (e.g., f32 to f32) */
    CUresult err = cuMemcpyDtoD((CUdeviceptr)dst, (CUdeviceptr)src, numel * sizeof(float));
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        fprintf(stderr, "launch_cast_kernel: cuMemcpyDtoD failed: %s\n", errStr);
    }
}

/* Seed the random number generator */
void sigil_cuda_randn_seed(uint64_t seed) {
    g_randn_state = seed;
}

/* ============================================================================
 * GEMM Implementation — Real GPU SGEMM via NVRTC
 * ============================================================================ */

/* SigilVec layout for Vec<f32>: len, capacity, float* data (packed 4 bytes/elem) */
typedef struct { int64_t len; int64_t capacity; float* data; } SigilVecF32;

/* ---- Tiled SGEMM NT kernel: C = A @ B^T
 *   A: [M x K] row-major
 *   B: [N x K] row-major (rows of B are treated as columns of B^T)
 *   C: [M x N] output
 *
 * Thread (ty,tx) in block (by,bx) computes C[by*16+ty][bx*16+tx].
 * Shared memory tiles: sA[ty][tx] = A[row][t*16+tx]
 *                      sB[ty][tx] = B[col][t*16+ty]  (col = bx*16+tx)
 * Inner product: sum_k sA[ty][k] * sB[k][tx]
 *              = sum_k A[row][t*16+k] * B[col][t*16+k]  ✓
 */
static const char* SGEMM_NT_SRC =
"#define TILE 16\n"
"extern \"C\" __global__ void sgemm_nt(\n"
"    const float* __restrict__ A,\n"
"    const float* __restrict__ B,\n"
"    float* __restrict__ C,\n"
"    int M, int N, int K) {\n"
"    __shared__ float sA[TILE][TILE], sB[TILE][TILE];\n"
"    int ty = threadIdx.y, tx = threadIdx.x;\n"
"    int row = blockIdx.y * TILE + ty;\n"
"    int col = blockIdx.x * TILE + tx;\n"
"    float acc = 0.0f;\n"
"    for (int t = 0; t * TILE < K; t++) {\n"
"        int aK = t * TILE + tx;\n"
"        int bK = t * TILE + ty;\n"
"        sA[ty][tx] = (row < M && aK < K) ? A[row * K + aK] : 0.0f;\n"
"        sB[ty][tx] = (col < N && bK < K) ? B[col * K + bK] : 0.0f;\n"
"        __syncthreads();\n"
"        for (int k = 0; k < TILE; k++) acc += sA[ty][k] * sB[k][tx];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < M && col < N) C[row * N + col] = acc;\n"
"}\n";

/* ---- Tiled SGEMM NN kernel: C = A @ B
 *   A: [M x K] row-major
 *   B: [K x N] row-major
 *   C: [M x N] output
 */
static const char* SGEMM_NN_SRC =
"#define TILE 16\n"
"extern \"C\" __global__ void sgemm_nn(\n"
"    const float* __restrict__ A,\n"
"    const float* __restrict__ B,\n"
"    float* __restrict__ C,\n"
"    int M, int N, int K) {\n"
"    __shared__ float sA[TILE][TILE], sB[TILE][TILE];\n"
"    int ty = threadIdx.y, tx = threadIdx.x;\n"
"    int row = blockIdx.y * TILE + ty;\n"
"    int col = blockIdx.x * TILE + tx;\n"
"    float acc = 0.0f;\n"
"    for (int t = 0; t * TILE < K; t++) {\n"
"        int aK = t * TILE + tx;\n"
"        int bK = t * TILE + ty;\n"
"        sA[ty][tx] = (row < M && aK < K) ? A[row * K + aK] : 0.0f;\n"
"        sB[ty][tx] = (bK < K && col < N) ? B[bK * N + col] : 0.0f;\n"
"        __syncthreads();\n"
"        for (int k = 0; k < TILE; k++) acc += sA[ty][k] * sB[k][tx];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < M && col < N) C[row * N + col] = acc;\n"
"}\n";

/* ---- Tiled SGEMM TN kernel: C = A^T @ B
 *   A: [K x M] row-major  (A^T is [M x K])
 *   B: [K x N] row-major
 *   C: [M x N] output
 * sA[ty][tx] = A[aK * M + row]  (transpose access: A^T[row, aK] = A[aK, row])
 * sB[ty][tx] = B[bK * N + col]  (same as NN)
 */
static const char* SGEMM_TN_SRC =
"#define TILE 16\n"
"extern \"C\" __global__ void sgemm_tn(\n"
"    const float* __restrict__ A,\n"
"    const float* __restrict__ B,\n"
"    float* __restrict__ C,\n"
"    int M, int N, int K) {\n"
"    __shared__ float sA[TILE][TILE], sB[TILE][TILE];\n"
"    int ty = threadIdx.y, tx = threadIdx.x;\n"
"    int row = blockIdx.y * TILE + ty;\n"
"    int col = blockIdx.x * TILE + tx;\n"
"    float acc = 0.0f;\n"
"    for (int t = 0; t * TILE < K; t++) {\n"
"        int aK = t * TILE + tx;\n"
"        int bK = t * TILE + ty;\n"
"        sA[ty][tx] = (row < M && aK < K) ? A[aK * M + row] : 0.0f;\n"
"        sB[ty][tx] = (bK < K && col < N) ? B[bK * N + col] : 0.0f;\n"
"        __syncthreads();\n"
"        for (int k = 0; k < TILE; k++) acc += sA[ty][k] * sB[k][tx];\n"
"        __syncthreads();\n"
"    }\n"
"    if (row < M && col < N) C[row * N + col] = acc;\n"
"}\n";

static CUfunction g_sgemm_nt_fn = NULL;
static CUfunction g_sgemm_nn_fn = NULL;
static CUfunction g_sgemm_tn_fn = NULL;
static int g_sgemm_compiled = 0;

static int compile_sgemm_kernel(const char* src, const char* name, CUfunction* out_fn) {
    if (!g_cuda_initialized && !sigil_cuda_init()) return 0;

    nvrtcProgram prog;
    nvrtcResult nr = nvrtcCreateProgram(&prog, src, "sgemm.cu", 0, NULL, NULL);
    if (nr != NVRTC_SUCCESS) {
        fprintf(stderr, "nvrtcCreateProgram failed for %s: %s\n", name, nvrtcGetErrorString(nr));
        return 0;
    }

    /* Try SM89 (Ada) first, fall back to SM75 (Turing) */
    const char* opts89[] = {"--gpu-architecture=compute_89"};
    nr = nvrtcCompileProgram(prog, 1, opts89);
    if (nr != NVRTC_SUCCESS) {
        const char* opts75[] = {"--gpu-architecture=compute_75"};
        nr = nvrtcCompileProgram(prog, 1, opts75);
    }
    if (nr != NVRTC_SUCCESS) {
        size_t logSz;
        nvrtcGetProgramLogSize(prog, &logSz);
        char* log = (char*)malloc(logSz);
        nvrtcGetProgramLog(prog, log);
        fprintf(stderr, "NVRTC compile failed for %s:\n%s\n", name, log);
        free(log);
        nvrtcDestroyProgram(&prog);
        return 0;
    }

    size_t ptxSz;
    nvrtcGetPTXSize(prog, &ptxSz);
    char* ptx = (char*)malloc(ptxSz);
    nvrtcGetPTX(prog, ptx);
    nvrtcDestroyProgram(&prog);

    CUmodule mod;
    CUresult cr = cuModuleLoadDataEx(&mod, ptx, 0, NULL, NULL);
    free(ptx);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "cuModuleLoadDataEx failed for %s: %d\n", name, cr);
        return 0;
    }

    cr = cuModuleGetFunction(out_fn, mod, name);
    if (cr != CUDA_SUCCESS) {
        fprintf(stderr, "cuModuleGetFunction failed for %s: %d\n", name, cr);
        return 0;
    }
    return 1;
}

static int ensure_sgemm_kernels() {
    if (g_sgemm_compiled) return (g_sgemm_nt_fn != NULL);
    g_sgemm_compiled = 1;
    fprintf(stderr, "[CUDA] ensure_sgemm_kernels: g_cuda_initialized=%d\n", g_cuda_initialized);
    fprintf(stderr, "[CUDA] Compiling SGEMM kernels via NVRTC...\n");
    int ok_nt = compile_sgemm_kernel(SGEMM_NT_SRC, "sgemm_nt", &g_sgemm_nt_fn);
    int ok_nn = compile_sgemm_kernel(SGEMM_NN_SRC, "sgemm_nn", &g_sgemm_nn_fn);
    int ok_tn = compile_sgemm_kernel(SGEMM_TN_SRC, "sgemm_tn", &g_sgemm_tn_fn);
    if (ok_nt && ok_nn && ok_tn)
        fprintf(stderr, "[CUDA] SGEMM kernels compiled successfully (SM89/SM75).\n");
    else
        fprintf(stderr, "[CUDA] WARNING: SGEMM kernel compilation failed (nt=%d nn=%d tn=%d).\n", ok_nt, ok_nn, ok_tn);
    return ok_nt;
}

int64_t sigil_cuda_is_available(void) {
    int64_t r = sigil_cuda_init();
    fprintf(stderr, "[CUDA-DEBUG] sigil_cuda_is_available: init=%lld initialized=%d\n",
            (long long)r, g_cuda_initialized);
    return r;
}

/*
 * sigil_cuda_gemm_f32 - Matrix multiplication C = A @ B (device pointers)
 * Now uses actual GPU kernel instead of CPU fallback.
 */
int64_t sigil_cuda_gemm_f32(
    int64_t a_ptr, int64_t b_ptr, int64_t c_ptr,
    int64_t m, int64_t n, int64_t k
) {
    if (!ensure_sgemm_kernels() || !g_sgemm_nn_fn) {
        fprintf(stderr, "sigil_cuda_gemm_f32: kernel unavailable\n");
        return -1;
    }
    int M = (int)m, N = (int)n, K = (int)k;
    void* args[] = { &a_ptr, &b_ptr, &c_ptr, &M, &N, &K };
    int tile = 16;
    unsigned gx = ((unsigned)N + tile - 1) / tile;
    unsigned gy = ((unsigned)M + tile - 1) / tile;
    CUresult cr = cuLaunchKernel(g_sgemm_nn_fn, gx, gy, 1, tile, tile, 1, 0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "sigil_cuda_gemm_f32 launch failed: %s\n", s);
        return -1;
    }
    cuCtxSynchronize();
    return 0;
}

/*
 * sigil_cuda_sgemm_host_nt — Host-to-host SGEMM via GPU: C = A @ B^T
 *
 * Takes CPU Vec<f32> pointers for A [M×K] and B [N×K].
 * Uploads to GPU, runs tiled SGEMM kernel, downloads result.
 * Returns new Vec<f32> containing C [M×N].
 *
 * This is the main acceleration entry point for the training loop.
 */
SigilVecF32 sigil_cuda_sgemm_host_nt(SigilVecF32 a, SigilVecF32 b, int64_t M, int64_t N, int64_t K) {
    SigilVecF32 result = {0, 0, NULL};

    if (!ensure_sgemm_kernels() || !g_sgemm_nt_fn) {
        fprintf(stderr, "sigil_cuda_sgemm_host_nt: kernel unavailable\n");
        return result;
    }

    size_t a_bytes = (size_t)(M * K) * sizeof(float);
    size_t b_bytes = (size_t)(N * K) * sizeof(float);
    size_t c_bytes = (size_t)(M * N) * sizeof(float);

    /* Allocate device memory */
    CUdeviceptr d_a, d_b, d_c;
    if (cuMemAlloc(&d_a, a_bytes) != CUDA_SUCCESS) { fprintf(stderr, "sgemm_host_nt: alloc A failed\n"); return result; }
    if (cuMemAlloc(&d_b, b_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); fprintf(stderr, "sgemm_host_nt: alloc B failed\n"); return result; }
    if (cuMemAlloc(&d_c, c_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); cuMemFree(d_b); fprintf(stderr, "sgemm_host_nt: alloc C failed\n"); return result; }

    /* Upload A and B */
    cuMemcpyHtoD(d_a, a.data, a_bytes);
    cuMemcpyHtoD(d_b, b.data, b_bytes);

    /* Launch tiled SGEMM NT kernel */
    int iM = (int)M, iN = (int)N, iK = (int)K;
    void* args[] = { &d_a, &d_b, &d_c, &iM, &iN, &iK };
    unsigned tile = 16;
    unsigned gx = ((unsigned)iN + tile - 1) / tile;
    unsigned gy = ((unsigned)iM + tile - 1) / tile;
    CUresult cr = cuLaunchKernel(g_sgemm_nt_fn, gx, gy, 1, tile, tile, 1, 0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "sgemm_host_nt launch failed: %s\n", s);
        cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
        return result;
    }
    cuCtxSynchronize();

    /* Download C */
    float* c_host = (float*)malloc(c_bytes);
    if (!c_host) { cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c); return result; }
    cuMemcpyDtoH(c_host, d_c, c_bytes);

    cuMemFree(d_a);
    cuMemFree(d_b);
    cuMemFree(d_c);

    result.data = c_host;
    result.len = M * N;
    result.capacity = M * N;
    return result;
}

/*
 * sigil_cuda_sgemm_host_nn — Host-to-host SGEMM via GPU: C = A @ B
 *
 * A: [M×K], B: [K×N], C: [M×N]
 * Used in backward pass (gradient through weight).
 */
SigilVecF32 sigil_cuda_sgemm_host_nn(SigilVecF32 a, SigilVecF32 b, int64_t M, int64_t N, int64_t K) {
    SigilVecF32 result = {0, 0, NULL};

    if (!ensure_sgemm_kernels() || !g_sgemm_nn_fn) {
        fprintf(stderr, "sigil_cuda_sgemm_host_nn: kernel unavailable\n");
        return result;
    }

    size_t a_bytes = (size_t)(M * K) * sizeof(float);
    size_t b_bytes = (size_t)(K * N) * sizeof(float);
    size_t c_bytes = (size_t)(M * N) * sizeof(float);

    CUdeviceptr d_a, d_b, d_c;
    if (cuMemAlloc(&d_a, a_bytes) != CUDA_SUCCESS) return result;
    if (cuMemAlloc(&d_b, b_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); return result; }
    if (cuMemAlloc(&d_c, c_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); cuMemFree(d_b); return result; }

    cuMemcpyHtoD(d_a, a.data, a_bytes);
    cuMemcpyHtoD(d_b, b.data, b_bytes);

    int iM = (int)M, iN = (int)N, iK = (int)K;
    void* args[] = { &d_a, &d_b, &d_c, &iM, &iN, &iK };
    unsigned tile = 16;
    unsigned gx = ((unsigned)iN + tile - 1) / tile;
    unsigned gy = ((unsigned)iM + tile - 1) / tile;
    CUresult cr = cuLaunchKernel(g_sgemm_nn_fn, gx, gy, 1, tile, tile, 1, 0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) { cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c); return result; }
    cuCtxSynchronize();

    float* c_host = (float*)malloc(c_bytes);
    if (!c_host) { cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c); return result; }
    cuMemcpyDtoH(c_host, d_c, c_bytes);

    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);

    result.data = c_host;
    result.len = M * N;
    result.capacity = M * N;
    return result;
}

/* sigil_cuda_is_available is defined earlier in this file */

/*
 * sigil_cuda_sgemm_nt_fill — GPU SGEMM NT into pre-allocated output buffer.
 *
 * C = A @ B^T,  A:[M×K], B:[N×K], out:[M×N]
 * Returns M*N on success, 0 on failure.
 * Uses i64 return (not Vec) so Sigil LLVM codegen emits the call correctly.
 */
int64_t sigil_cuda_sgemm_nt_fill(SigilVecF32 a, SigilVecF32 b, SigilVecF32 out, int64_t M, int64_t N, int64_t K) {
    if (!ensure_sgemm_kernels() || !g_sgemm_nt_fn) {
        fprintf(stderr, "sgemm_nt_fill: kernel unavailable\n");
        return 0;
    }
    if (!a.data || !b.data || !out.data || out.len < M * N) {
        fprintf(stderr, "sgemm_nt_fill: bad input (a=%p b=%p out=%p out.len=%ld need=%ld)\n",
                (void*)a.data, (void*)b.data, (void*)out.data, (long)out.len, (long)(M*N));
        return 0;
    }

    size_t a_bytes = (size_t)(M * K) * sizeof(float);
    size_t b_bytes = (size_t)(N * K) * sizeof(float);
    size_t c_bytes = (size_t)(M * N) * sizeof(float);

    CUdeviceptr d_a, d_b, d_c;
    if (cuMemAlloc(&d_a, a_bytes) != CUDA_SUCCESS) return 0;
    if (cuMemAlloc(&d_b, b_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); return 0; }
    if (cuMemAlloc(&d_c, c_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); cuMemFree(d_b); return 0; }

    cuMemcpyHtoD(d_a, a.data, a_bytes);
    cuMemcpyHtoD(d_b, b.data, b_bytes);

    int iM = (int)M, iN = (int)N, iK = (int)K;
    void* args[] = { &d_a, &d_b, &d_c, &iM, &iN, &iK };
    unsigned tile = 16;
    unsigned gx = ((unsigned)iN + tile - 1) / tile;
    unsigned gy = ((unsigned)iM + tile - 1) / tile;
    CUresult cr = cuLaunchKernel(g_sgemm_nt_fn, gx, gy, 1, tile, tile, 1, 0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "sgemm_nt_fill launch failed: %s\n", s);
        cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
        return 0;
    }
    cuCtxSynchronize();

    cuMemcpyDtoH(out.data, d_c, c_bytes);
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    return M * N;
}

/* ============================================================================
 * Sigil Vec<f32> native ABI SGEMM helpers
 *
 * SIGIL VEC ABI: Vec<f32> is passed as a SINGLE int64_t (pointer to SigilVec).
 *   SigilVec: { int64_t len, int64_t capacity, int64_t* data }
 *   Elements: data[i] = (int64_t)(uint32_t)(f32_bits) — float bits zero-extended.
 *
 * sigil_sgemm_nt_sv / sigil_sgemm_nn_sv:
 *   - All args are int64_t, fitting exactly 6 integer registers (rdi..r9)
 *   - a, b, out are SigilVec* pointers (Sigil passes Vec<f32> as single i64)
 *   - Unpacks i64→float[], runs GPU SGEMM, packs float[]→i64 into out
 * ============================================================================ */

typedef struct { int64_t len; int64_t capacity; int64_t* data; } SigilVecNative;

static float* sigil_unpack_vec(int64_t vec_ptr, int64_t expected_n) {
    SigilVecNative* v = (SigilVecNative*)(uintptr_t)vec_ptr;
    if (!v || !v->data || v->len < expected_n) {
        fprintf(stderr, "sigil_unpack_vec: bad ptr=%p len=%ld need=%ld\n",
                (void*)v, v ? (long)v->len : -1L, (long)expected_n);
        return NULL;
    }
    float* buf = (float*)malloc((size_t)expected_n * sizeof(float));
    if (!buf) return NULL;
    for (int64_t i = 0; i < expected_n; i++) {
        uint32_t bits = (uint32_t)(v->data[i] & 0xFFFFFFFFULL);
        memcpy(&buf[i], &bits, sizeof(float));
    }
    return buf;
}

static void sigil_pack_vec(int64_t vec_ptr, const float* src, int64_t n) {
    SigilVecNative* v = (SigilVecNative*)(uintptr_t)vec_ptr;
    if (!v || !v->data) return;
    for (int64_t i = 0; i < n; i++) {
        uint32_t bits;
        memcpy(&bits, &src[i], sizeof(float));
        v->data[i] = (int64_t)(uint64_t)bits;
    }
}

int64_t sigil_sgemm_nt_sv(int64_t a_ptr, int64_t b_ptr, int64_t out_ptr,
                           int64_t M, int64_t N, int64_t K) {
    if (!ensure_sgemm_kernels() || !g_sgemm_nt_fn) return 0;
    float* a_f = sigil_unpack_vec(a_ptr, M * K);
    float* b_f = sigil_unpack_vec(b_ptr, N * K);
    if (!a_f || !b_f) { free(a_f); free(b_f); return 0; }
    size_t a_bytes = (size_t)(M * K) * sizeof(float);
    size_t b_bytes = (size_t)(N * K) * sizeof(float);
    size_t c_bytes = (size_t)(M * N) * sizeof(float);
    float* c_f = (float*)malloc(c_bytes);
    if (!c_f) { free(a_f); free(b_f); return 0; }
    CUdeviceptr d_a, d_b, d_c;
    if (cuMemAlloc(&d_a, a_bytes) != CUDA_SUCCESS) { free(a_f); free(b_f); free(c_f); return 0; }
    if (cuMemAlloc(&d_b, b_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); free(a_f); free(b_f); free(c_f); return 0; }
    if (cuMemAlloc(&d_c, c_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); cuMemFree(d_b); free(a_f); free(b_f); free(c_f); return 0; }
    cuMemcpyHtoD(d_a, a_f, a_bytes);
    cuMemcpyHtoD(d_b, b_f, b_bytes);
    free(a_f); free(b_f);
    int iM = (int)M, iN = (int)N, iK = (int)K;
    void* args[] = { &d_a, &d_b, &d_c, &iM, &iN, &iK };
    unsigned tile = 16;
    CUresult cr = cuLaunchKernel(g_sgemm_nt_fn,
        ((unsigned)iN+tile-1)/tile, ((unsigned)iM+tile-1)/tile, 1,
        tile, tile, 1, 0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "sgemm_nt_sv launch failed: %s\n", s);
        cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c); free(c_f); return 0;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(c_f, d_c, c_bytes);
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    sigil_pack_vec(out_ptr, c_f, M * N);
    free(c_f);
    return M * N;
}

/* sigil_sgemm_tn_sv: C[M,N] = A^T @ B
 * A is [K, M] row-major (i.e., A stored as [m, k] where M=k, K=m in caller),
 * B is [K, N] row-major, C is [M, N].
 * Caller: matmul_AT_vecs(a[m,k], b[m,n]) → result[k,n]
 *   → sigil_sgemm_tn_sv(a, b, out, k, n, m)
 */
int64_t sigil_sgemm_tn_sv(int64_t a_ptr, int64_t b_ptr, int64_t out_ptr,
                           int64_t M, int64_t N, int64_t K) {
    if (!ensure_sgemm_kernels() || !g_sgemm_tn_fn) return 0;
    float* a_f = sigil_unpack_vec(a_ptr, M * K);
    float* b_f = sigil_unpack_vec(b_ptr, N * K);
    if (!a_f || !b_f) { free(a_f); free(b_f); return 0; }
    size_t a_bytes = (size_t)(M * K) * sizeof(float);
    size_t b_bytes = (size_t)(N * K) * sizeof(float);
    size_t c_bytes = (size_t)(M * N) * sizeof(float);
    float* c_f = (float*)malloc(c_bytes);
    if (!c_f) { free(a_f); free(b_f); return 0; }
    CUdeviceptr d_a, d_b, d_c;
    if (cuMemAlloc(&d_a, a_bytes) != CUDA_SUCCESS) { free(a_f); free(b_f); free(c_f); return 0; }
    if (cuMemAlloc(&d_b, b_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); free(a_f); free(b_f); free(c_f); return 0; }
    if (cuMemAlloc(&d_c, c_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); cuMemFree(d_b); free(a_f); free(b_f); free(c_f); return 0; }
    cuMemcpyHtoD(d_a, a_f, a_bytes);
    cuMemcpyHtoD(d_b, b_f, b_bytes);
    free(a_f); free(b_f);
    int iM = (int)M, iN = (int)N, iK = (int)K;
    void* args[] = { &d_a, &d_b, &d_c, &iM, &iN, &iK };
    unsigned tile = 16;
    CUresult cr = cuLaunchKernel(g_sgemm_tn_fn,
        ((unsigned)iN+tile-1)/tile, ((unsigned)iM+tile-1)/tile, 1,
        tile, tile, 1, 0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "sgemm_tn_sv launch failed: %s\n", s);
        cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c); free(c_f); return 0;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(c_f, d_c, c_bytes);
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    sigil_pack_vec(out_ptr, c_f, M * N);
    free(c_f);
    return M * N;
}

int64_t sigil_sgemm_nn_sv(int64_t a_ptr, int64_t b_ptr, int64_t out_ptr,
                           int64_t M, int64_t N, int64_t K) {
    if (!ensure_sgemm_kernels() || !g_sgemm_nn_fn) return 0;
    float* a_f = sigil_unpack_vec(a_ptr, M * K);
    float* b_f = sigil_unpack_vec(b_ptr, K * N);
    if (!a_f || !b_f) { free(a_f); free(b_f); return 0; }
    size_t a_bytes = (size_t)(M * K) * sizeof(float);
    size_t b_bytes = (size_t)(K * N) * sizeof(float);
    size_t c_bytes = (size_t)(M * N) * sizeof(float);
    float* c_f = (float*)malloc(c_bytes);
    if (!c_f) { free(a_f); free(b_f); return 0; }
    CUdeviceptr d_a, d_b, d_c;
    if (cuMemAlloc(&d_a, a_bytes) != CUDA_SUCCESS) { free(a_f); free(b_f); free(c_f); return 0; }
    if (cuMemAlloc(&d_b, b_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); free(a_f); free(b_f); free(c_f); return 0; }
    if (cuMemAlloc(&d_c, c_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); cuMemFree(d_b); free(a_f); free(b_f); free(c_f); return 0; }
    cuMemcpyHtoD(d_a, a_f, a_bytes);
    cuMemcpyHtoD(d_b, b_f, b_bytes);
    free(a_f); free(b_f);
    int iM = (int)M, iN = (int)N, iK = (int)K;
    void* args[] = { &d_a, &d_b, &d_c, &iM, &iN, &iK };
    unsigned tile = 16;
    CUresult cr = cuLaunchKernel(g_sgemm_nn_fn,
        ((unsigned)iN+tile-1)/tile, ((unsigned)iM+tile-1)/tile, 1,
        tile, tile, 1, 0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "sgemm_nn_sv launch failed: %s\n", s);
        cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c); free(c_f); return 0;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(c_f, d_c, c_bytes);
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    sigil_pack_vec(out_ptr, c_f, M * N);
    free(c_f);
    return M * N;
}

/*
 * sigil_vec_data_i64 — Extract raw data pointer from Vec<f32> as i64.
 *
 * Takes ONE Vec<f32> arg (fits in 3 integer registers — known to work in Sigil ABI).
 * Used so Sigil code can get a raw pointer to pass to all-i64-arg SGEMM functions.
 */
int64_t sigil_vec_data_i64(SigilVecF32 v) {
    return (int64_t)(uintptr_t)v.data;
}

/*
 * sigil_sgemm_nt_raw — GPU SGEMM NT via raw pointers (all i64 args).
 *
 * C = A @ B^T,  A:[M×K], B:[N×K], out:[M×N]
 * All arguments are i64 — no struct passing, works with Sigil LLVM ABI.
 * Returns M*N on success, 0 on failure.
 */
int64_t sigil_sgemm_nt_raw(int64_t a_ptr, int64_t b_ptr, int64_t out_ptr,
                            int64_t M, int64_t N, int64_t K) {
    if (!ensure_sgemm_kernels() || !g_sgemm_nt_fn) return 0;
    float* a = (float*)(uintptr_t)a_ptr;
    float* b = (float*)(uintptr_t)b_ptr;
    float* out = (float*)(uintptr_t)out_ptr;
    if (!a || !b || !out) return 0;

    size_t a_bytes = (size_t)(M * K) * sizeof(float);
    size_t b_bytes = (size_t)(N * K) * sizeof(float);
    size_t c_bytes = (size_t)(M * N) * sizeof(float);

    CUdeviceptr d_a, d_b, d_c;
    if (cuMemAlloc(&d_a, a_bytes) != CUDA_SUCCESS) return 0;
    if (cuMemAlloc(&d_b, b_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); return 0; }
    if (cuMemAlloc(&d_c, c_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); cuMemFree(d_b); return 0; }

    cuMemcpyHtoD(d_a, a, a_bytes);
    cuMemcpyHtoD(d_b, b, b_bytes);

    int iM = (int)M, iN = (int)N, iK = (int)K;
    void* args[] = { &d_a, &d_b, &d_c, &iM, &iN, &iK };
    unsigned tile = 16;
    unsigned gx = ((unsigned)iN + tile - 1) / tile;
    unsigned gy = ((unsigned)iM + tile - 1) / tile;
    CUresult cr = cuLaunchKernel(g_sgemm_nt_fn, gx, gy, 1, tile, tile, 1, 0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "sgemm_nt_raw launch failed: %s\n", s);
        cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
        return 0;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(out, d_c, c_bytes);
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    return M * N;
}

/*
 * sigil_sgemm_nn_raw — GPU SGEMM NN via raw pointers (all i64 args).
 *
 * C = A @ B,  A:[M×K], B:[K×N], out:[M×N]
 */
int64_t sigil_sgemm_nn_raw(int64_t a_ptr, int64_t b_ptr, int64_t out_ptr,
                            int64_t M, int64_t N, int64_t K) {
    if (!ensure_sgemm_kernels() || !g_sgemm_nn_fn) return 0;
    float* a = (float*)(uintptr_t)a_ptr;
    float* b = (float*)(uintptr_t)b_ptr;
    float* out = (float*)(uintptr_t)out_ptr;
    if (!a || !b || !out) return 0;

    size_t a_bytes = (size_t)(M * K) * sizeof(float);
    size_t b_bytes = (size_t)(K * N) * sizeof(float);
    size_t c_bytes = (size_t)(M * N) * sizeof(float);

    CUdeviceptr d_a, d_b, d_c;
    if (cuMemAlloc(&d_a, a_bytes) != CUDA_SUCCESS) return 0;
    if (cuMemAlloc(&d_b, b_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); return 0; }
    if (cuMemAlloc(&d_c, c_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); cuMemFree(d_b); return 0; }

    cuMemcpyHtoD(d_a, a, a_bytes);
    cuMemcpyHtoD(d_b, b, b_bytes);

    int iM = (int)M, iN = (int)N, iK = (int)K;
    void* args[] = { &d_a, &d_b, &d_c, &iM, &iN, &iK };
    unsigned tile = 16;
    unsigned gx = ((unsigned)iN + tile - 1) / tile;
    unsigned gy = ((unsigned)iM + tile - 1) / tile;
    CUresult cr = cuLaunchKernel(g_sgemm_nn_fn, gx, gy, 1, tile, tile, 1, 0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "sgemm_nn_raw launch failed: %s\n", s);
        cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
        return 0;
    }
    cuCtxSynchronize();
    cuMemcpyDtoH(out, d_c, c_bytes);
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    return M * N;
}

/*
 * sigil_cuda_sgemm_nn_fill — GPU SGEMM NN into pre-allocated output buffer.
 *
 * C = A @ B,  A:[M×K], B:[K×N], out:[M×N]
 * Returns M*N on success, 0 on failure.
 */
int64_t sigil_cuda_sgemm_nn_fill(SigilVecF32 a, SigilVecF32 b, SigilVecF32 out, int64_t M, int64_t N, int64_t K) {
    if (!ensure_sgemm_kernels() || !g_sgemm_nn_fn) {
        fprintf(stderr, "sgemm_nn_fill: kernel unavailable\n");
        return 0;
    }
    if (!a.data || !b.data || !out.data || out.len < M * N) {
        fprintf(stderr, "sgemm_nn_fill: bad input\n");
        return 0;
    }

    size_t a_bytes = (size_t)(M * K) * sizeof(float);
    size_t b_bytes = (size_t)(K * N) * sizeof(float);
    size_t c_bytes = (size_t)(M * N) * sizeof(float);

    CUdeviceptr d_a, d_b, d_c;
    if (cuMemAlloc(&d_a, a_bytes) != CUDA_SUCCESS) return 0;
    if (cuMemAlloc(&d_b, b_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); return 0; }
    if (cuMemAlloc(&d_c, c_bytes) != CUDA_SUCCESS) { cuMemFree(d_a); cuMemFree(d_b); return 0; }

    cuMemcpyHtoD(d_a, a.data, a_bytes);
    cuMemcpyHtoD(d_b, b.data, b_bytes);

    int iM = (int)M, iN = (int)N, iK = (int)K;
    void* args[] = { &d_a, &d_b, &d_c, &iM, &iN, &iK };
    unsigned tile = 16;
    unsigned gx = ((unsigned)iN + tile - 1) / tile;
    unsigned gy = ((unsigned)iM + tile - 1) / tile;
    CUresult cr = cuLaunchKernel(g_sgemm_nn_fn, gx, gy, 1, tile, tile, 1, 0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "sgemm_nn_fill launch failed: %s\n", s);
        cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
        return 0;
    }
    cuCtxSynchronize();

    cuMemcpyDtoH(out.data, d_c, c_bytes);
    cuMemFree(d_a); cuMemFree(d_b); cuMemFree(d_c);
    return M * N;
}

/* ============================================================================
 * Nihil forward-pass CUDA kernels (device-pointer interface)
 *
 * All functions take raw int64_t device pointers (from StoragePtr.data_ptr).
 * No host copies — data stays on GPU throughout the forward pass.
 * ============================================================================ */

static const char* EMBED_GATHER_SRC =
"extern \"C\" __global__ void embed_gather(\n"
"    float* out, const int* ids, const float* wte,\n"
"    int batch_seq, int d_model) {\n"
"    int tok = blockIdx.x;\n"
"    if (tok >= batch_seq) return;\n"
"    int vocab_row = ids[tok];\n"
"    int d = threadIdx.x;\n"
"    while (d < d_model) {\n"
"        out[tok * d_model + d] = wte[vocab_row * d_model + d];\n"
"        d += blockDim.x;\n"
"    }\n"
"}\n";

static const char* RMSNORM_FWD_SRC =
"extern \"C\" __global__ void rmsnorm_fwd(\n"
"    float* out, const float* x, const float* w,\n"
"    int rows, int d_model, float eps) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= rows) return;\n"
"    const float* xr = x + row * d_model;\n"
"    float* or_ = out + row * d_model;\n"
"    __shared__ float shared[256];\n"
"    int tid = threadIdx.x;\n"
"    float ss = 0.0f;\n"
"    for (int i = tid; i < d_model; i += blockDim.x) ss += xr[i] * xr[i];\n"
"    shared[tid] = ss;\n"
"    __syncthreads();\n"
"    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n"
"        if (tid < s) shared[tid] += shared[tid + s];\n"
"        __syncthreads();\n"
"    }\n"
"    float rms = rsqrtf(shared[0] / (float)d_model + eps);\n"
"    for (int i = tid; i < d_model; i += blockDim.x)\n"
"        or_[i] = xr[i] * rms * w[i];\n"
"}\n";

static const char* ADD_INTO_SRC =
"extern \"C\" __global__ void add_into(\n"
"    float* out, const float* a, const float* b, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i < n) out[i] = a[i] + b[i];\n"
"}\n";

static const char* SWIGLU_FWD_SRC =
"extern \"C\" __global__ void swiglu(\n"
"    float* out, const float* gate, const float* up, int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float g = gate[i];\n"
"    float silu = g / (1.0f + expf(-g));\n"
"    out[i] = silu * up[i];\n"
"}\n";

static const char* CE_LOSS_SRC =
"extern \"C\" __global__ void ce_loss_fwd(\n"
"    float* out_loss, const float* logits, const int* targets,\n"
"    int batch_seq, int vocab) {\n"
"    int row = blockIdx.x;\n"
"    if (row >= batch_seq) return;\n"
"    const float* row_logits = logits + row * vocab;\n"
"    int tgt = targets[row];\n"
"    float mx = row_logits[0];\n"
"    for (int i = 1; i < vocab; i++) if (row_logits[i] > mx) mx = row_logits[i];\n"
"    float sum_exp = 0.0f;\n"
"    for (int i = 0; i < vocab; i++) sum_exp += expf(row_logits[i] - mx);\n"
"    float log_prob = row_logits[tgt] - mx - logf(sum_exp);\n"
"    atomicAdd(out_loss, -log_prob / (float)batch_seq);\n"
"}\n";

static CUfunction g_embed_gather_fn  = NULL;
static CUfunction g_rmsnorm_fwd_fn   = NULL;
static CUfunction g_add_into_fn      = NULL;
static CUfunction g_swiglu_fwd_fn    = NULL;
static CUfunction g_ce_loss_fwd_fn   = NULL;
static int        g_nihil_kernels_compiled = 0;

static int ensure_nihil_kernels(void) {
    if (g_nihil_kernels_compiled) return (g_embed_gather_fn != NULL);
    g_nihil_kernels_compiled = 1;
    if (!ensure_sgemm_kernels()) return 0;
    compile_sgemm_kernel(EMBED_GATHER_SRC, "embed_gather",  &g_embed_gather_fn);
    compile_sgemm_kernel(RMSNORM_FWD_SRC,  "rmsnorm_fwd",   &g_rmsnorm_fwd_fn);
    compile_sgemm_kernel(ADD_INTO_SRC,     "add_into",      &g_add_into_fn);
    compile_sgemm_kernel(SWIGLU_FWD_SRC,   "swiglu",        &g_swiglu_fwd_fn);
    compile_sgemm_kernel(CE_LOSS_SRC,      "ce_loss_fwd",   &g_ce_loss_fwd_fn);
    int ok = (g_embed_gather_fn && g_rmsnorm_fwd_fn && g_add_into_fn
              && g_swiglu_fwd_fn && g_ce_loss_fwd_fn);
    fprintf(stderr, "[CUDA] Nihil fwd kernels: %s\n", ok ? "OK" : "PARTIAL/FAILED");
    return ok;
}

void sigil_cuda_embed_gather_f32(
    int64_t out_ptr, int64_t ids_ptr, int64_t wte_ptr,
    int64_t batch_seq, int64_t d_model)
{
    if (!ensure_nihil_kernels() || !g_embed_gather_fn) return;
    int bs = (int)batch_seq, dm = (int)d_model;
    void* args[] = { &out_ptr, &ids_ptr, &wte_ptr, &bs, &dm };
    int threads = (dm < 256) ? dm : 256;
    cuLaunchKernel(g_embed_gather_fn, (unsigned)bs, 1, 1, threads, 1, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();
}

void sigil_cuda_rmsnorm_f32(
    int64_t out_ptr, int64_t x_ptr, int64_t w_ptr,
    int64_t rows, int64_t d_model, int64_t eps_bits)
{
    if (!ensure_nihil_kernels() || !g_rmsnorm_fwd_fn) return;
    float eps; memcpy(&eps, &eps_bits, sizeof(float));
    int r = (int)rows, dm = (int)d_model;
    void* args[] = { &out_ptr, &x_ptr, &w_ptr, &r, &dm, &eps };
    int threads = (dm < 256) ? ((dm + 31) & ~31) : 256;
    cuLaunchKernel(g_rmsnorm_fwd_fn, (unsigned)r, 1, 1, threads, 1, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();
}

void sigil_cuda_add_f32(int64_t out_ptr, int64_t a_ptr, int64_t b_ptr, int64_t n) {
    if (!ensure_nihil_kernels() || !g_add_into_fn) return;
    int in = (int)n;
    void* args[] = { &out_ptr, &a_ptr, &b_ptr, &in };
    unsigned blocks = ((unsigned)n + 255) / 256;
    cuLaunchKernel(g_add_into_fn, blocks, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();
}

void sigil_cuda_swiglu_f32(int64_t out_ptr, int64_t gate_ptr, int64_t up_ptr, int64_t n) {
    if (!ensure_nihil_kernels() || !g_swiglu_fwd_fn) return;
    int in = (int)n;
    void* args[] = { &out_ptr, &gate_ptr, &up_ptr, &in };
    unsigned blocks = ((unsigned)n + 255) / 256;
    cuLaunchKernel(g_swiglu_fwd_fn, blocks, 1, 1, 256, 1, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();
}

void sigil_cuda_ce_loss_f32(
    int64_t out_loss_ptr, int64_t logits_ptr, int64_t targets_ptr,
    int64_t batch_seq, int64_t vocab)
{
    if (!ensure_nihil_kernels() || !g_ce_loss_fwd_fn) return;
    cuMemsetD8((CUdeviceptr)out_loss_ptr, 0, sizeof(float));
    int bs = (int)batch_seq, v = (int)vocab;
    void* args[] = { &out_loss_ptr, &logits_ptr, &targets_ptr, &bs, &v };
    cuLaunchKernel(g_ce_loss_fwd_fn, (unsigned)bs, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();
}

void sigil_cuda_sgemm_nt_raw(
    int64_t a_ptr, int64_t b_ptr, int64_t c_ptr,
    int64_t M, int64_t N, int64_t K)
{
    if (!ensure_sgemm_kernels() || !g_sgemm_nt_fn) return;
    int iM = (int)M, iN = (int)N, iK = (int)K;
    void* args[] = { &a_ptr, &b_ptr, &c_ptr, &iM, &iN, &iK };
    int tile = 16;
    unsigned gx = ((unsigned)iN + tile - 1) / tile;
    unsigned gy = ((unsigned)iM + tile - 1) / tile;
    cuLaunchKernel(g_sgemm_nt_fn, gx, gy, 1, tile, tile, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();
}

void sigil_cuda_sgemm_nn_raw(
    int64_t a_ptr, int64_t b_ptr, int64_t c_ptr,
    int64_t M, int64_t N, int64_t K)
{
    if (!ensure_sgemm_kernels() || !g_sgemm_nn_fn) return;
    int iM = (int)M, iN = (int)N, iK = (int)K;
    void* args[] = { &a_ptr, &b_ptr, &c_ptr, &iM, &iN, &iK };
    int tile = 16;
    unsigned gx = ((unsigned)iN + tile - 1) / tile;
    unsigned gy = ((unsigned)iM + tile - 1) / tile;
    cuLaunchKernel(g_sgemm_nn_fn, gx, gy, 1, tile, tile, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();
}

void sigil_cuda_sgemm_tn_raw(
    int64_t a_ptr, int64_t b_ptr, int64_t c_ptr,
    int64_t M, int64_t N, int64_t K)
{
    if (!ensure_sgemm_kernels() || !g_sgemm_tn_fn) return;
    int iM = (int)M, iN = (int)N, iK = (int)K;
    void* args[] = { &a_ptr, &b_ptr, &c_ptr, &iM, &iN, &iK };
    int tile = 16;
    unsigned gx = ((unsigned)iN + tile - 1) / tile;
    unsigned gy = ((unsigned)iM + tile - 1) / tile;
    cuLaunchKernel(g_sgemm_tn_fn, gx, gy, 1, tile, tile, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();
}

/* ============================================================================
 * Causal MHA kernel + utility functions for Nihil forward pass
 * ============================================================================ */

/* Causal multi-head self-attention.
 * q, k, v: [batch*seq, heads*head_dim]  (contiguous; heads packed per token)
 * out:     [batch*seq, heads*head_dim]
 * Grid: (batch*heads,)  Block: (seq,)
 * Each thread handles one query token within its (batch, head) pair.
 */
static const char* ATTN_CAUSAL_SRC =
"extern \"C\" __global__ void attn_causal(\n"
"    float* out, const float* q, const float* k, const float* v,\n"
"    int batch, int seq, int heads, int hd) {\n"
"    int bh = blockIdx.x;\n"
"    int b  = bh / heads;\n"
"    int h  = bh % heads;\n"
"    int s  = threadIdx.x;\n"
"    if (b >= batch || s >= seq) return;\n"
"    int d_model = heads * hd;\n"
"    float scale = rsqrtf((float)hd);\n"
"    /* query vector for this (b, s, h) */\n"
"    const float* q_s = q + (b * seq + s) * d_model + h * hd;\n"
"    /* compute causal attention scores */\n"
"    float scores[512];\n"
"    float max_score = -1e9f;\n"
"    for (int t = 0; t <= s; t++) {\n"
"        const float* k_t = k + (b * seq + t) * d_model + h * hd;\n"
"        float dot = 0.0f;\n"
"        for (int d = 0; d < hd; d++) dot += q_s[d] * k_t[d];\n"
"        scores[t] = dot * scale;\n"
"        if (scores[t] > max_score) max_score = scores[t];\n"
"    }\n"
"    for (int t = s + 1; t < seq; t++) scores[t] = -1e9f;\n"
"    /* softmax */\n"
"    float sum_exp = 0.0f;\n"
"    for (int t = 0; t < seq; t++) {\n"
"        scores[t] = expf(scores[t] - max_score);\n"
"        sum_exp += scores[t];\n"
"    }\n"
"    float inv_sum = 1.0f / sum_exp;\n"
"    for (int t = 0; t < seq; t++) scores[t] *= inv_sum;\n"
"    /* weighted V sum */\n"
"    float* out_s = out + (b * seq + s) * d_model + h * hd;\n"
"    for (int d = 0; d < hd; d++) {\n"
"        float val = 0.0f;\n"
"        for (int t = 0; t < seq; t++) {\n"
"            const float* v_t = v + (b * seq + t) * d_model + h * hd;\n"
"            val += scores[t] * v_t[d];\n"
"        }\n"
"        out_s[d] = val;\n"
"    }\n"
"}\n";

static CUfunction g_attn_causal_fn = NULL;

/* Extend ensure_nihil_kernels to also compile the attention kernel.
 * Called lazily on first use of any Nihil fwd kernel. */
static int ensure_nihil_attn_kernel(void) {
    if (g_attn_causal_fn) return 1;
    if (!ensure_nihil_kernels()) return 0;   /* ensures SGEMM + other kernels */
    compile_sgemm_kernel(ATTN_CAUSAL_SRC, "attn_causal", &g_attn_causal_fn);
    if (!g_attn_causal_fn) {
        fprintf(stderr, "[CUDA] attn_causal kernel compile FAILED\n");
        return 0;
    }
    fprintf(stderr, "[CUDA] attn_causal kernel: OK\n");
    return 1;
}

void sigil_cuda_attn_fwd_f32(
    int64_t out_ptr, int64_t q_ptr, int64_t k_ptr, int64_t v_ptr,
    int64_t batch, int64_t seq, int64_t heads, int64_t hd)
{
    if (!ensure_nihil_attn_kernel() || !g_attn_causal_fn) return;
    int ib = (int)batch, is = (int)seq, ih = (int)heads, ihd = (int)hd;
    void* args[] = { &out_ptr, &q_ptr, &k_ptr, &v_ptr, &ib, &is, &ih, &ihd };
    unsigned blocks = (unsigned)(batch * heads);
    unsigned threads = (unsigned)seq;
    CUresult cr = cuLaunchKernel(g_attn_causal_fn,
                                  blocks, 1, 1,
                                  threads, 1, 1,
                                  0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "[CUDA] attn_causal launch failed: %s\n", s);
    }
    cuCtxSynchronize();
}

/* ============================================================================
 * Attention forward-store + backward kernels
 *
 * attn_causal_store: identical to attn_causal but also writes softmax
 *   probabilities to probs[B*H, S, S] for use by the backward pass.
 *
 * attn_causal_bwd: computes dQ, dK, dV from saved probabilities.
 *   Thread (bh, s) owns query position s and:
 *     1. Loads P[bh, s, 0..s] from probs buffer
 *     2. Computes dP[t] = dot(dOut[s], V[t])  for t <= s
 *     3. Computes dp_sum = sum_t P[t]*dP[t]  (scalar softmax correction)
 *     4. Overwrites dP[t] = P[t]*(dP[t] - dp_sum)*scale  (this is dScore[t])
 *     5. Writes dQ[s] = sum_t dScore[t]*K[t]  (no race — one thread per s)
 *     6. atomicAdd: dK[t] += dScore[t]*Q[s], dV[t] += P[t]*dOut[s]
 *
 * Constraints (same as attn_causal):
 *   seq <= 512  (local dP array is float[512])
 *   threads per block = seq  (must be <= 1024)
 *   grid = batch * heads blocks
 * ============================================================================ */

static const char* ATTN_CAUSAL_STORE_SRC =
"extern \"C\" __global__ void attn_causal_store(\n"
"    float* out, float* probs,\n"
"    const float* q, const float* k, const float* v,\n"
"    int batch, int seq, int heads, int hd) {\n"
"    int bh = blockIdx.x;\n"
"    int b  = bh / heads;\n"
"    int h  = bh % heads;\n"
"    int s  = threadIdx.x;\n"
"    if (b >= batch || s >= seq) return;\n"
"    int d_model = heads * hd;\n"
"    float scale = rsqrtf((float)hd);\n"
"    const float* q_s = q + (b * seq + s) * d_model + h * hd;\n"
"    /* causal attention scores */\n"
"    float scores[512];\n"
"    float max_score = -1e9f;\n"
"    for (int t = 0; t <= s; t++) {\n"
"        const float* k_t = k + (b * seq + t) * d_model + h * hd;\n"
"        float dot = 0.0f;\n"
"        for (int d = 0; d < hd; d++) dot += q_s[d] * k_t[d];\n"
"        scores[t] = dot * scale;\n"
"        if (scores[t] > max_score) max_score = scores[t];\n"
"    }\n"
"    for (int t = s + 1; t < seq; t++) scores[t] = -1e9f;\n"
"    /* softmax */\n"
"    float sum_exp = 0.0f;\n"
"    for (int t = 0; t < seq; t++) {\n"
"        scores[t] = expf(scores[t] - max_score);\n"
"        sum_exp += scores[t];\n"
"    }\n"
"    float inv_sum = 1.0f / sum_exp;\n"
"    for (int t = 0; t < seq; t++) scores[t] *= inv_sum;\n"
"    /* write probs[bh, s, 0..seq] */\n"
"    float* probs_row = probs + (long long)bh * seq * seq + (long long)s * seq;\n"
"    for (int t = 0; t < seq; t++) probs_row[t] = scores[t];\n"
"    /* weighted V sum -> out */\n"
"    float* out_s = out + (b * seq + s) * d_model + h * hd;\n"
"    for (int d = 0; d < hd; d++) {\n"
"        float val = 0.0f;\n"
"        for (int t = 0; t < seq; t++) {\n"
"            const float* v_t = v + (b * seq + t) * d_model + h * hd;\n"
"            val += scores[t] * v_t[d];\n"
"        }\n"
"        out_s[d] = val;\n"
"    }\n"
"}\n";

static const char* ATTN_CAUSAL_BWD_SRC =
"extern \"C\" __global__ void attn_causal_bwd(\n"
"    float* dq, float* dk, float* dv,\n"
"    const float* probs,\n"
"    const float* q, const float* k, const float* v,\n"
"    const float* dout,\n"
"    int batch, int seq, int heads, int hd) {\n"
"    int bh = blockIdx.x;\n"
"    int b  = bh / heads;\n"
"    int h  = bh % heads;\n"
"    int s  = threadIdx.x;   /* query position */\n"
"    if (b >= batch || s >= seq) return;\n"
"    int d_model = heads * hd;\n"
"    float scale = rsqrtf((float)hd);\n"
"    /* pointers for this query token */\n"
"    const float* P_row  = probs + (long long)bh * seq * seq + (long long)s * seq;\n"
"    const float* dout_s = dout  + (b * seq + s) * d_model + h * hd;\n"
"    const float* q_s    = q     + (b * seq + s) * d_model + h * hd;\n"
"    /* step 1: dP[t] = dot(dOut[s], V[t])  for t <= s */\n"
"    float dP[512];\n"
"    for (int t = 0; t <= s; t++) {\n"
"        const float* v_t = v + (b * seq + t) * d_model + h * hd;\n"
"        float dot = 0.0f;\n"
"        for (int d = 0; d < hd; d++) dot += dout_s[d] * v_t[d];\n"
"        dP[t] = dot;\n"
"    }\n"
"    /* step 2: dp_sum = sum_t P[t]*dP[t] */\n"
"    float dp_sum = 0.0f;\n"
"    for (int t = 0; t <= s; t++) dp_sum += P_row[t] * dP[t];\n"
"    /* step 3: dScore[t] = P[t]*(dP[t] - dp_sum)*scale  (overwrite dP) */\n"
"    for (int t = 0; t <= s; t++)\n"
"        dP[t] = P_row[t] * (dP[t] - dp_sum) * scale;\n"
"    /* step 4: dQ[s] = sum_t dScore[t]*K[t]  (direct write, no race) */\n"
"    float* dq_s = dq + (b * seq + s) * d_model + h * hd;\n"
"    for (int d = 0; d < hd; d++) {\n"
"        float val = 0.0f;\n"
"        for (int t = 0; t <= s; t++) {\n"
"            const float* k_t = k + (b * seq + t) * d_model + h * hd;\n"
"            val += dP[t] * k_t[d];\n"
"        }\n"
"        dq_s[d] = val;\n"
"    }\n"
"    /* step 5: atomicAdd dK[t] += dScore[t]*Q[s], dV[t] += P[t]*dOut[s] */\n"
"    for (int t = 0; t <= s; t++) {\n"
"        float* dk_t = dk + (b * seq + t) * d_model + h * hd;\n"
"        float* dv_t = dv + (b * seq + t) * d_model + h * hd;\n"
"        float ds = dP[t];          /* dScore[t] */\n"
"        float pt = P_row[t];       /* P[t]      */\n"
"        for (int d = 0; d < hd; d++) {\n"
"            atomicAdd(&dk_t[d], ds * q_s[d]);\n"
"            atomicAdd(&dv_t[d], pt * dout_s[d]);\n"
"        }\n"
"    }\n"
"}\n";

static CUfunction g_attn_store_fn = NULL;
static CUfunction g_attn_bwd_fn   = NULL;

static int ensure_attn_backward_kernels(void) {
    if (g_attn_store_fn && g_attn_bwd_fn) return 1;
    /* ensure_nihil_attn_kernel also ensures sgemm + other nihil kernels */
    if (!ensure_nihil_attn_kernel()) return 0;
    if (!g_attn_store_fn) {
        compile_sgemm_kernel(ATTN_CAUSAL_STORE_SRC, "attn_causal_store", &g_attn_store_fn);
        if (!g_attn_store_fn) {
            fprintf(stderr, "[CUDA] attn_causal_store kernel compile FAILED\n");
            return 0;
        }
        fprintf(stderr, "[CUDA] attn_causal_store kernel: OK\n");
    }
    if (!g_attn_bwd_fn) {
        compile_sgemm_kernel(ATTN_CAUSAL_BWD_SRC, "attn_causal_bwd", &g_attn_bwd_fn);
        if (!g_attn_bwd_fn) {
            fprintf(stderr, "[CUDA] attn_causal_bwd kernel compile FAILED\n");
            return 0;
        }
        fprintf(stderr, "[CUDA] attn_causal_bwd kernel: OK\n");
    }
    return 1;
}

/* Forward pass + save probabilities.
 * out_ptr:   [batch*seq, heads*hd]  f32 — attention output
 * probs_ptr: [batch*heads, seq, seq] f32 — softmax probs (caller pre-allocates) */
void sigil_cuda_attn_fwd_store_f32(
    int64_t out_ptr, int64_t probs_ptr,
    int64_t q_ptr, int64_t k_ptr, int64_t v_ptr,
    int64_t batch, int64_t seq, int64_t heads, int64_t hd)
{
    if (!ensure_attn_backward_kernels() || !g_attn_store_fn) return;
    int ib = (int)batch, is = (int)seq, ih = (int)heads, ihd = (int)hd;
    void* args[] = { &out_ptr, &probs_ptr, &q_ptr, &k_ptr, &v_ptr,
                     &ib, &is, &ih, &ihd };
    unsigned blocks  = (unsigned)(batch * heads);
    unsigned threads = (unsigned)seq;
    CUresult cr = cuLaunchKernel(g_attn_store_fn,
                                  blocks, 1, 1,
                                  threads, 1, 1,
                                  0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "[CUDA] attn_causal_store launch failed: %s\n", s);
    }
    cuCtxSynchronize();
}

/* Backward pass.
 * dq_ptr, dk_ptr, dv_ptr: gradient buffers [batch*seq, heads*hd] f32
 *   dK and dV must be pre-zeroed by caller (atomicAdd accumulates into them).
 *   dQ is written directly (no pre-zero needed).
 * probs_ptr: saved from attn_fwd_store_f32 for the same Q/K/V */
void sigil_cuda_attn_bwd_f32(
    int64_t dq_ptr, int64_t dk_ptr, int64_t dv_ptr,
    int64_t probs_ptr,
    int64_t q_ptr, int64_t k_ptr, int64_t v_ptr,
    int64_t dout_ptr,
    int64_t batch, int64_t seq, int64_t heads, int64_t hd)
{
    if (!ensure_attn_backward_kernels() || !g_attn_bwd_fn) return;
    int ib = (int)batch, is = (int)seq, ih = (int)heads, ihd = (int)hd;
    void* args[] = { &dq_ptr, &dk_ptr, &dv_ptr, &probs_ptr,
                     &q_ptr, &k_ptr, &v_ptr, &dout_ptr,
                     &ib, &is, &ih, &ihd };
    unsigned blocks  = (unsigned)(batch * heads);
    unsigned threads = (unsigned)seq;
    CUresult cr = cuLaunchKernel(g_attn_bwd_fn,
                                  blocks, 1, 1,
                                  threads, 1, 1,
                                  0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "[CUDA] attn_causal_bwd launch failed: %s\n", s);
    }
    cuCtxSynchronize();
}

/* ============================================================================
 * Wave 2: Elementwise backward kernels
 *
 * ce_backward:      d_logits = (softmax(logits) - one_hot(targets)) / batch_seq
 * rmsnorm_backward: dx, dw via standard RMSNorm backward
 * swiglu_backward:  d_gate, d_up elementwise from saved gate/up
 * embed_scatter:    d_wte[ids[t],:] += d_hidden[t,:]  (atomicAdd for safe multi-write)
 * ============================================================================ */

static const char* CE_BWD_SRC =
"extern \"C\" __global__ void ce_backward_f32(\n"
"    float* d_logits, const float* logits, const long long* targets,\n"
"    int batch_seq, int vocab) {\n"
"    int t = blockIdx.x;\n"
"    if (t >= batch_seq) return;\n"
"    const float* row = logits + (long long)t * vocab;\n"
"    float* drow = d_logits + (long long)t * vocab;\n"
"    /* numerically-stable softmax */\n"
"    float max_v = row[0];\n"
"    for (int v = 1; v < vocab; v++) if (row[v] > max_v) max_v = row[v];\n"
"    float Z = 0.0f;\n"
"    for (int v = 0; v < vocab; v++) Z += expf(row[v] - max_v);\n"
"    float inv_Z  = 1.0f / Z;\n"
"    float scale  = 1.0f / (float)batch_seq;\n"
"    int   tgt    = (int)targets[t];\n"
"    for (int v = 0; v < vocab; v++) {\n"
"        float p = expf(row[v] - max_v) * inv_Z;\n"
"        drow[v] = (p - (v == tgt ? 1.0f : 0.0f)) * scale;\n"
"    }\n"
"}\n";

static const char* RMSNORM_BWD_SRC =
"extern \"C\" __global__ void rmsnorm_bwd_f32(\n"
"    float* dx, float* dw,\n"
"    const float* dout, const float* x, const float* w,\n"
"    int rows, int d_model, float eps) {\n"
"    int r = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (r >= rows) return;\n"
"    const float* xr  = x    + r * d_model;\n"
"    const float* dor = dout + r * d_model;\n"
"    float* dxr = dx + r * d_model;\n"
"    /* compute rms */\n"
"    float sum_sq = 0.0f;\n"
"    for (int d = 0; d < d_model; d++) sum_sq += xr[d] * xr[d];\n"
"    float rms      = sqrtf(sum_sq / (float)d_model + eps);\n"
"    float rms_r    = 1.0f / rms;\n"
"    /* dot(dout .* w, norm) where norm = x * rms_r */\n"
"    float dot_term = 0.0f;\n"
"    for (int d = 0; d < d_model; d++) dot_term += dor[d] * w[d] * xr[d] * rms_r;\n"
"    /* dx[r,d] = rms_r * (dout[r,d]*w[d] - norm[r,d]/d_model * dot_term) */\n"
"    for (int d = 0; d < d_model; d++)\n"
"        dxr[d] = rms_r * (dor[d] * w[d] - xr[d] * rms_r / (float)d_model * dot_term);\n"
"    /* dw[d] += dout[r,d] * norm[r,d] */\n"
"    for (int d = 0; d < d_model; d++)\n"
"        atomicAdd(&dw[d], dor[d] * xr[d] * rms_r);\n"
"}\n";

static const char* SWIGLU_BWD_SRC =
"extern \"C\" __global__ void swiglu_bwd_f32(\n"
"    float* d_gate, float* d_up,\n"
"    const float* d_out, const float* gate, const float* up,\n"
"    int n) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float g   = gate[i];\n"
"    float u   = up[i];\n"
"    float dof = d_out[i];\n"
"    float sig = 1.0f / (1.0f + expf(-g));\n"
"    d_up[i]   = dof * g * sig;\n"
"    d_gate[i] = dof * u * sig * (1.0f + g * (1.0f - sig));\n"
"}\n";

static const char* EMBED_SCATTER_SRC =
"extern \"C\" __global__ void embed_scatter_f32(\n"
"    float* d_wte, const long long* ids, const float* d_hidden,\n"
"    int batch_seq, int d_model) {\n"
"    int t = blockIdx.x;\n"
"    int d = threadIdx.x;\n"
"    if (t >= batch_seq || d >= d_model) return;\n"
"    long long id = ids[t];\n"
"    atomicAdd(&d_wte[id * d_model + d], d_hidden[(long long)t * d_model + d]);\n"
"}\n";

static CUfunction g_ce_bwd_fn       = NULL;
static CUfunction g_rmsnorm_bwd_fn  = NULL;
static CUfunction g_swiglu_bwd_fn   = NULL;
static CUfunction g_embed_scatter_fn= NULL;

static int ensure_wave2_kernels(void) {
    if (g_ce_bwd_fn && g_rmsnorm_bwd_fn && g_swiglu_bwd_fn && g_embed_scatter_fn) return 1;
    if (!ensure_nihil_kernels()) return 0;
    if (!g_ce_bwd_fn)
        compile_sgemm_kernel(CE_BWD_SRC,       "ce_backward_f32",   &g_ce_bwd_fn);
    if (!g_rmsnorm_bwd_fn)
        compile_sgemm_kernel(RMSNORM_BWD_SRC,  "rmsnorm_bwd_f32",   &g_rmsnorm_bwd_fn);
    if (!g_swiglu_bwd_fn)
        compile_sgemm_kernel(SWIGLU_BWD_SRC,   "swiglu_bwd_f32",    &g_swiglu_bwd_fn);
    if (!g_embed_scatter_fn)
        compile_sgemm_kernel(EMBED_SCATTER_SRC,"embed_scatter_f32", &g_embed_scatter_fn);
    return (g_ce_bwd_fn && g_rmsnorm_bwd_fn && g_swiglu_bwd_fn && g_embed_scatter_fn) ? 1 : 0;
}

/* CE backward: d_logits = (softmax(logits) - one_hot(targets)) / batch_seq
 * d_logits is written directly — no pre-zero required.
 * targets_ptr: device pointer to int64_t array of length batch_seq. */
void sigil_cuda_ce_backward_f32(
    int64_t d_logits_ptr, int64_t logits_ptr, int64_t targets_ptr,
    int64_t batch_seq, int64_t vocab)
{
    if (!ensure_wave2_kernels() || !g_ce_bwd_fn) return;
    int ibs = (int)batch_seq, ivc = (int)vocab;
    void* args[] = { &d_logits_ptr, &logits_ptr, &targets_ptr, &ibs, &ivc };
    CUresult cr = cuLaunchKernel(g_ce_bwd_fn,
                                  (unsigned)batch_seq, 1, 1,
                                  1, 1, 1,
                                  0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "[CUDA] ce_backward launch failed: %s\n", s);
    }
    cuCtxSynchronize();
}

/* RMSNorm backward.
 * dx: written per-row, no pre-zero needed.
 * dw: accumulates via atomicAdd — must be pre-zeroed by caller.
 * eps_bits: f32 epsilon as int64 (e.g. 0x358637BD = 1e-6). */
void sigil_cuda_rmsnorm_backward_f32(
    int64_t dx_ptr, int64_t dw_ptr,
    int64_t dout_ptr, int64_t x_ptr, int64_t w_ptr,
    int64_t rows, int64_t d_model, int64_t eps_bits)
{
    if (!ensure_wave2_kernels() || !g_rmsnorm_bwd_fn) return;
    int32_t lo = (int32_t)(eps_bits & 0xFFFFFFFFLL);
    float eps; memcpy(&eps, &lo, sizeof(float));
    int ir = (int)rows, idm = (int)d_model;
    void* args[] = { &dx_ptr, &dw_ptr, &dout_ptr, &x_ptr, &w_ptr, &ir, &idm, &eps };
    unsigned threads = 256;
    unsigned blocks  = ((unsigned)rows + threads - 1) / threads;
    CUresult cr = cuLaunchKernel(g_rmsnorm_bwd_fn,
                                  blocks, 1, 1,
                                  threads, 1, 1,
                                  0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "[CUDA] rmsnorm_bwd launch failed: %s\n", s);
    }
    cuCtxSynchronize();
}

/* SwiGLU backward: d_gate and d_up are written directly (elementwise, no race). */
void sigil_cuda_swiglu_backward_f32(
    int64_t d_gate_ptr, int64_t d_up_ptr,
    int64_t d_out_ptr, int64_t gate_ptr, int64_t up_ptr,
    int64_t n)
{
    if (!ensure_wave2_kernels() || !g_swiglu_bwd_fn) return;
    int in = (int)n;
    void* args[] = { &d_gate_ptr, &d_up_ptr, &d_out_ptr, &gate_ptr, &up_ptr, &in };
    unsigned threads = 256;
    unsigned blocks  = ((unsigned)n + threads - 1) / threads;
    CUresult cr = cuLaunchKernel(g_swiglu_bwd_fn,
                                  blocks, 1, 1,
                                  threads, 1, 1,
                                  0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "[CUDA] swiglu_bwd launch failed: %s\n", s);
    }
    cuCtxSynchronize();
}

/* Embed scatter: d_wte[ids[t],:] += d_hidden[t,:]
 * d_wte must be pre-zeroed. ids_ptr: device pointer to int64_t token IDs.
 * Grid = (batch_seq,), Block = (d_model,): one block per token, one thread per dim. */
void sigil_cuda_embed_scatter_f32(
    int64_t d_wte_ptr, int64_t ids_ptr, int64_t d_hidden_ptr,
    int64_t batch_seq, int64_t d_model)
{
    if (!ensure_wave2_kernels() || !g_embed_scatter_fn) return;
    int ibs = (int)batch_seq, idm = (int)d_model;
    void* args[] = { &d_wte_ptr, &ids_ptr, &d_hidden_ptr, &ibs, &idm };
    CUresult cr = cuLaunchKernel(g_embed_scatter_fn,
                                  (unsigned)batch_seq, 1, 1,
                                  (unsigned)d_model, 1, 1,
                                  0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "[CUDA] embed_scatter launch failed: %s\n", s);
    }
    cuCtxSynchronize();
}

/* Fill device buffer with a constant f32 value (value_bits = f32 bits as int64). */
void sigil_cuda_fill_const_f32(int64_t device_ptr, int64_t n, int64_t val_bits) {
    float val;
    int32_t lo = (int32_t)(val_bits & 0xFFFFFFFFLL);
    memcpy(&val, &lo, sizeof(float));
    size_t bytes = (size_t)n * sizeof(float);
    float* host = (float*)malloc(bytes);
    if (!host) { fprintf(stderr, "[CUDA] fill_const: malloc failed\n"); return; }
    for (size_t i = 0; i < (size_t)n; i++) host[i] = val;
    cuMemcpyHtoD((CUdeviceptr)device_ptr, host, bytes);
    free(host);
}

/* Copy 1 f32 from device to host and print as "step=N loss=F" */
void sigil_cuda_print_loss(int64_t loss_ptr, int64_t step) {
    float val = 0.0f;
    CUresult r = cuMemcpyDtoH(&val, (CUdeviceptr)loss_ptr, sizeof(float));
    if (r != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(r, &s);
        fprintf(stderr, "[CUDA] print_loss copy failed: %s\n", s);
        val = -1.0f;
    }
    printf("[STEP %lld] loss=%.4f\n", (long long)step, (double)val);
    fflush(stdout);
}

/* ============================================================================
 * Wave 3: GPU AdamW optimizer
 *
 * adamw_step: in-place AdamW update on device pointers.
 *   m[i]  ← β1*m[i] + (1−β1)*g[i]
 *   v[i]  ← β2*v[i] + (1−β2)*g[i]²
 *   m_hat = m[i] / (1 − β1^step)
 *   v_hat = v[i] / (1 − β2^step)
 *   w[i]  ← w[i] * (1−lr*wd) − lr * m_hat / (sqrt(v_hat) + eps)
 *
 * All hyperparameters are f32 bits packed into int64 (same ABI as fill_const).
 * step is 1-based: step=1 for first optimizer call.
 * ============================================================================ */

static const char* ADAMW_SRC =
"extern \"C\" __global__ void adamw_step_f32(\n"
"    float* w, float* g, float* m, float* v,\n"
"    int n, int step,\n"
"    float lr, float wd, float b1, float b2, float eps) {\n"
"    int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
"    if (i >= n) return;\n"
"    float gi = g[i];\n"
"    float mi = b1 * m[i] + (1.0f - b1) * gi;\n"
"    float vi = b2 * v[i] + (1.0f - b2) * gi * gi;\n"
"    m[i] = mi;\n"
"    v[i] = vi;\n"
"    float bias1  = 1.0f - powf(b1, (float)step);\n"
"    float bias2  = 1.0f - powf(b2, (float)step);\n"
"    float m_hat  = mi / bias1;\n"
"    float v_hat  = vi / bias2;\n"
"    w[i] = w[i] * (1.0f - lr * wd) - lr * m_hat / (sqrtf(v_hat) + eps);\n"
"}\n";

static CUfunction g_adamw_fn = NULL;

static int ensure_adamw_kernel(void) {
    if (g_adamw_fn) return 1;
    if (!ensure_nihil_kernels()) return 0;
    compile_sgemm_kernel(ADAMW_SRC, "adamw_step_f32", &g_adamw_fn);
    fprintf(stderr, "[CUDA] adamw kernel: %s\n", g_adamw_fn ? "OK" : "FAILED");
    return g_adamw_fn ? 1 : 0;
}

/* In-place AdamW step. step is 1-based (first call uses step=1).
 * All hyperparams are f32 bits packed in int64_t. */
void sigil_cuda_adamw_step_f32(
    int64_t w_ptr, int64_t g_ptr, int64_t m_ptr, int64_t v_ptr,
    int64_t n, int64_t step,
    int64_t lr_bits, int64_t wd_bits,
    int64_t b1_bits, int64_t b2_bits, int64_t eps_bits)
{
    if (!ensure_adamw_kernel() || !g_adamw_fn) return;
    /* Extract f32 hyperparams from bit-packed int64 */
    float lr, wd, b1, b2, eps;
    int32_t lo;
    lo = (int32_t)(lr_bits  & 0xFFFFFFFFLL); memcpy(&lr,  &lo, 4);
    lo = (int32_t)(wd_bits  & 0xFFFFFFFFLL); memcpy(&wd,  &lo, 4);
    lo = (int32_t)(b1_bits  & 0xFFFFFFFFLL); memcpy(&b1,  &lo, 4);
    lo = (int32_t)(b2_bits  & 0xFFFFFFFFLL); memcpy(&b2,  &lo, 4);
    lo = (int32_t)(eps_bits & 0xFFFFFFFFLL); memcpy(&eps, &lo, 4);
    int in = (int)n, istep = (int)step;
    void* args[] = { &w_ptr, &g_ptr, &m_ptr, &v_ptr,
                     &in, &istep, &lr, &wd, &b1, &b2, &eps };
    unsigned threads = 256;
    unsigned blocks  = ((unsigned)n + threads - 1) / threads;
    CUresult cr = cuLaunchKernel(g_adamw_fn,
                                  blocks, 1, 1,
                                  threads, 1, 1,
                                  0, NULL, args, NULL);
    if (cr != CUDA_SUCCESS) {
        const char* s; cuGetErrorString(cr, &s);
        fprintf(stderr, "[CUDA] adamw_step launch failed: %s\n", s);
    }
    /* No sync here — caller must sync if needed (avoids stalling between steps) */
}

/* ============================================================================
 * Test helper utilities — host-side assertion functions for Agent-TDD
 *
 * All functions copy device data to host and perform checks.
 * Return conventions: 0 = pass, non-zero = count of failures/violations.
 * tol_bits arguments: absolute tolerance encoded as f32 bits in int64_t
 *   (same ABI as sigil_cuda_fill_const_f32).
 * ============================================================================ */

/* Compare two device f32 buffers element-wise.
 * Returns count of elements where |a[i] - b[i]| > tol or either is non-finite. */
int64_t sigil_cuda_test_close(int64_t a_ptr, int64_t b_ptr, int64_t n, int64_t tol_bits) {
    int32_t lo = (int32_t)(tol_bits & 0xFFFFFFFFLL);
    float tol; memcpy(&tol, &lo, sizeof(float));
    size_t bytes = (size_t)n * sizeof(float);
    float *a = (float*)malloc(bytes);
    float *b = (float*)malloc(bytes);
    if (!a || !b) { free(a); free(b); return -1; }
    cuMemcpyDtoH(a, (CUdeviceptr)a_ptr, bytes);
    cuMemcpyDtoH(b, (CUdeviceptr)b_ptr, bytes);
    int64_t mismatches = 0;
    for (int64_t i = 0; i < n; i++) {
        float diff = a[i] - b[i]; if (diff < 0.f) diff = -diff;
        if (diff > tol || !isfinite(a[i]) || !isfinite(b[i])) {
            if (mismatches < 4)
                fprintf(stderr, "[TEST_CLOSE] mismatch[%lld]: a=%.6g b=%.6g diff=%.6g tol=%.6g\n",
                        (long long)i, (double)a[i], (double)b[i], (double)diff, (double)tol);
            mismatches++;
        }
    }
    free(a); free(b);
    return mismatches;
}

/* Check probs[BH, S, S] rows sum to 1.0 and all values are non-negative.
 * probs layout: row-major [BH, S, S] — outer index is (batch*heads).
 * tol_bits: tolerance for |row_sum - 1.0| as f32 bits.
 * Returns 0 = pass, 1 = fail (prints first few violations to stderr). */
int64_t sigil_cuda_test_probs_sum1(int64_t probs_ptr, int64_t BH, int64_t S, int64_t tol_bits) {
    int32_t lo = (int32_t)(tol_bits & 0xFFFFFFFFLL);
    float tol; memcpy(&tol, &lo, sizeof(float));
    int64_t total = BH * S * S;
    float *p = (float*)malloc((size_t)total * sizeof(float));
    if (!p) return 1;
    cuMemcpyDtoH(p, (CUdeviceptr)probs_ptr, (size_t)total * sizeof(float));
    int64_t fail = 0;
    for (int64_t bh = 0; bh < BH; bh++) {
        for (int64_t s = 0; s < S; s++) {
            float row_sum = 0.0f;
            for (int64_t t = 0; t < S; t++) {
                float v = p[bh*S*S + s*S + t];
                if (v < -tol) {
                    if (fail < 4)
                        fprintf(stderr, "[TEST_PROBS] negative: bh=%lld s=%lld t=%lld val=%.6g\n",
                                (long long)bh, (long long)s, (long long)t, (double)v);
                    fail = 1;
                }
                row_sum += v;
            }
            float err = row_sum - 1.0f; if (err < 0.f) err = -err;
            if (err > tol) {
                fprintf(stderr, "[TEST_PROBS] row_sum: bh=%lld s=%lld sum=%.6g (want 1.0, tol=%.6g)\n",
                        (long long)bh, (long long)s, (double)row_sum, (double)tol);
                fail = 1;
            }
        }
    }
    free(p);
    return fail;
}

/* Check causal mask: probs[bh, s, t] must equal 0 for all t > s.
 * Returns count of violations (0 = pass). */
int64_t sigil_cuda_test_probs_causal(int64_t probs_ptr, int64_t BH, int64_t S) {
    int64_t total = BH * S * S;
    float *p = (float*)malloc((size_t)total * sizeof(float));
    if (!p) return 1;
    cuMemcpyDtoH(p, (CUdeviceptr)probs_ptr, (size_t)total * sizeof(float));
    int64_t violations = 0;
    for (int64_t bh = 0; bh < BH; bh++) {
        for (int64_t s = 0; s < S; s++) {
            for (int64_t t = s + 1; t < S; t++) {
                float v = p[bh*S*S + s*S + t];
                if (v > 1e-7f) {
                    if (violations < 4)
                        fprintf(stderr, "[TEST_CAUSAL] future non-zero: bh=%lld s=%lld t=%lld val=%.6g\n",
                                (long long)bh, (long long)s, (long long)t, (double)v);
                    violations++;
                }
            }
        }
    }
    free(p);
    return violations;
}

/* Check all n f32 values are finite (no NaN or Inf).
 * Returns count of non-finite values (0 = pass). */
int64_t sigil_cuda_test_finite(int64_t ptr, int64_t n) {
    float *h = (float*)malloc((size_t)n * sizeof(float));
    if (!h) return 1;
    cuMemcpyDtoH(h, (CUdeviceptr)ptr, (size_t)n * sizeof(float));
    int64_t bad = 0;
    for (int64_t i = 0; i < n; i++) {
        if (!isfinite(h[i])) {
            if (bad < 4)
                fprintf(stderr, "[TEST_FINITE] non-finite[%lld]=%.6g\n",
                        (long long)i, (double)h[i]);
            bad++;
        }
    }
    free(h);
    return bad;
}

/* ============================================================================
 * Gradient-check helpers
 * ============================================================================ */

/* Read device f32[idx] — returned as i64 holding the raw 32-bit bit pattern. */
int64_t sigil_cuda_getf32(int64_t ptr, int64_t idx) {
    float v = 0.0f;
    cuMemcpyDtoH(&v, (CUdeviceptr)((char*)NULL + ptr + (size_t)idx * 4), sizeof(float));
    uint32_t bits;
    memcpy(&bits, &v, sizeof(float));
    return (int64_t)bits;
}

/* Write f32 (from 32-bit pattern in low bits of val_bits) to device f32[idx]. */
void sigil_cuda_setf32(int64_t ptr, int64_t idx, int64_t val_bits) {
    int32_t lo32 = (int32_t)(val_bits & 0xFFFFFFFFLL);
    float v;
    memcpy(&v, &lo32, sizeof(float));
    cuMemcpyHtoD((CUdeviceptr)((char*)NULL + ptr + (size_t)idx * 4), &v, sizeof(float));
}

/* Float add of two f32 bit-patterns — returns result as i64 bit-pattern. */
int64_t sigil_cuda_f32_add(int64_t a_bits, int64_t b_bits) {
    int32_t la = (int32_t)(a_bits & 0xFFFFFFFFLL);
    int32_t lb = (int32_t)(b_bits & 0xFFFFFFFFLL);
    float a, b;
    memcpy(&a, &la, sizeof(float));
    memcpy(&b, &lb, sizeof(float));
    float c = a + b;
    uint32_t bits;
    memcpy(&bits, &c, sizeof(float));
    return (int64_t)bits;
}

/* Float sub of two f32 bit-patterns — returns result as i64 bit-pattern. */
int64_t sigil_cuda_f32_sub(int64_t a_bits, int64_t b_bits) {
    int32_t la = (int32_t)(a_bits & 0xFFFFFFFFLL);
    int32_t lb = (int32_t)(b_bits & 0xFFFFFFFFLL);
    float a, b;
    memcpy(&a, &la, sizeof(float));
    memcpy(&b, &lb, sizeof(float));
    float c = a - b;
    uint32_t bits;
    memcpy(&bits, &c, sizeof(float));
    return (int64_t)bits;
}

/* Finite-difference gradient check.
 *   g_bits:     analytical gradient (f32 bit-pattern)
 *   lp_bits:    L(w + delta) loss (f32 bit-pattern)
 *   lm_bits:    L(w - delta) loss (f32 bit-pattern)
 *   delta_bits: perturbation magnitude (f32 bit-pattern)
 *
 * Computes:
 *   fd_grad  = (L+ - L-) / (2 * delta)
 *   rel_err  = |g - fd_grad| / (|g| + |fd_grad| + 1e-7)
 *
 * Prints diagnostics and returns 0 (pass, rel_err < 5%) or 1 (fail). */
int64_t sigil_cuda_fd_check(int64_t g_bits, int64_t lp_bits, int64_t lm_bits, int64_t delta_bits) {
    int32_t lo;
    float g, lp, lm, delta;
    lo = (int32_t)(g_bits     & 0xFFFFFFFFLL); memcpy(&g,     &lo, sizeof(float));
    lo = (int32_t)(lp_bits    & 0xFFFFFFFFLL); memcpy(&lp,    &lo, sizeof(float));
    lo = (int32_t)(lm_bits    & 0xFFFFFFFFLL); memcpy(&lm,    &lo, sizeof(float));
    lo = (int32_t)(delta_bits & 0xFFFFFFFFLL); memcpy(&delta, &lo, sizeof(float));

    float fd_grad = (lp - lm) / (2.0f * delta);
    float abs_err = fabsf(g - fd_grad);
    float denom   = fabsf(g) + fabsf(fd_grad) + 1e-7f;
    float rel_err = abs_err / denom;

    fprintf(stderr, "[GRAD_CHECK] g_analytical=%.6g  fd_grad=%.6g  rel_err=%.4f\n",
            (double)g, (double)fd_grad, (double)rel_err);
    fprintf(stderr, "[GRAD_CHECK] L+=%g  L-=%g  delta=%g  diff=%g\n",
            (double)lp, (double)lm, (double)delta, (double)(lp - lm));

    return (rel_err < 0.05f) ? 0 : 1;
}

/* Check that at least one of the n f32 values is non-zero and finite.
 * Returns 0 if any non-zero finite value is found (pass),
 *         1 if all values are zero or non-finite (fail — gradients not computed). */
int64_t sigil_cuda_test_nonzero(int64_t ptr, int64_t n) {
    float *h = (float*)malloc((size_t)n * sizeof(float));
    if (!h) return 1;
    cuMemcpyDtoH(h, (CUdeviceptr)ptr, (size_t)n * sizeof(float));
    int64_t found = 0;
    for (int64_t i = 0; i < n; i++) {
        if (h[i] != 0.0f && isfinite(h[i])) { found = 1; break; }
    }
    free(h);
    return found ? 0 : 1;
}
