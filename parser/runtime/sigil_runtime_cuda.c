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
