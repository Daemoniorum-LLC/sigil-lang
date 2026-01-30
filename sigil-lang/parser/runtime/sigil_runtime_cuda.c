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
