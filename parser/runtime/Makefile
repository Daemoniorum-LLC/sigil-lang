# Sigil Runtime Library Makefile
#
# Builds the runtime into a static library for faster AOT linking
# Supports LTO for maximum performance
# Supports TLS/OpenSSL for HTTPS connectivity

CC ?= clang
AR ?= ar
CFLAGS = -O3 -fPIC -Wall -Wextra

# LTO-enabled flags (works with both gcc and clang)
LTO_CFLAGS = -O3 -fPIC -Wall -Wextra -flto

# TLS flags
TLS_CFLAGS = -DSIGIL_TLS_SUPPORT
TLS_LIBS = -lssl -lcrypto

# CUDA flags
NVCC ?= nvcc
CUDA_CFLAGS = -O3 -I/usr/local/cuda/include
CUDA_LIBS = -lcuda -lnvrtc

# Output files
STATIC_LIB = libsigil_runtime.a
LTO_LIB = libsigil_runtime_lto.a
TLS_LIB = libsigil_runtime_tls.a
CUDA_LIB = libsigil_runtime_cuda.a
OBJECT = sigil_runtime.o
LTO_OBJECT = sigil_runtime_lto.o
TLS_OBJECT = sigil_runtime_tls.o
CUDA_OBJECT = sigil_runtime_cuda.o

.PHONY: all lto tls cuda clean help

all: $(STATIC_LIB)

lto: $(LTO_LIB)

tls: $(TLS_LIB)

cuda: $(CUDA_LIB)

help:
	@echo "Sigil Runtime Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all   - Build standard runtime library (default)"
	@echo "  lto   - Build with Link-Time Optimization"
	@echo "  tls   - Build with TLS/OpenSSL support"
	@echo "  clean - Remove all built files"
	@echo ""
	@echo "TLS requires OpenSSL development libraries:"
	@echo "  Ubuntu/Debian: apt install libssl-dev"
	@echo "  Fedora/RHEL:   dnf install openssl-devel"
	@echo "  macOS:         brew install openssl"

$(OBJECT): sigil_runtime.c
	$(CC) $(CFLAGS) -c sigil_runtime.c -o $(OBJECT)

$(LTO_OBJECT): sigil_runtime.c
	$(CC) $(LTO_CFLAGS) -c sigil_runtime.c -o $(LTO_OBJECT)

$(TLS_OBJECT): sigil_runtime.c
	$(CC) $(CFLAGS) $(TLS_CFLAGS) -c sigil_runtime.c -o $(TLS_OBJECT)

$(STATIC_LIB): $(OBJECT)
	$(AR) rcs $(STATIC_LIB) $(OBJECT)
	@echo "Built $(STATIC_LIB)"

$(LTO_LIB): $(LTO_OBJECT)
	gcc-ar rcs $(LTO_LIB) $(LTO_OBJECT) 2>/dev/null || $(AR) rcs $(LTO_LIB) $(LTO_OBJECT)
	@echo "Built $(LTO_LIB) (with LTO)"

$(TLS_LIB): $(TLS_OBJECT)
	$(AR) rcs $(TLS_LIB) $(TLS_OBJECT)
	@echo "Built $(TLS_LIB) (with TLS/OpenSSL support)"
	@echo "Link with: $(TLS_LIBS)"

# CUDA runtime - base runtime compiled with CUDA support + CUDA extensions
CUDA_BASE_OBJECT = sigil_runtime_cuda_base.o

$(CUDA_BASE_OBJECT): sigil_runtime.c
	$(CC) $(CFLAGS) -DSIGIL_CUDA_SUPPORT -c sigil_runtime.c -o $(CUDA_BASE_OBJECT)

$(CUDA_OBJECT): sigil_runtime_cuda.c
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) -c sigil_runtime_cuda.c -o $(CUDA_OBJECT)

$(CUDA_LIB): $(CUDA_BASE_OBJECT) $(CUDA_OBJECT)
	$(AR) rcs $(CUDA_LIB) $(CUDA_BASE_OBJECT) $(CUDA_OBJECT)
	@echo "Built $(CUDA_LIB) (with CUDA/Nihil support)"
	@echo "Link with: $(CUDA_LIBS)"

clean:
	rm -f $(OBJECT) $(LTO_OBJECT) $(TLS_OBJECT) $(CUDA_OBJECT) $(STATIC_LIB) $(LTO_LIB) $(TLS_LIB) $(CUDA_LIB)
