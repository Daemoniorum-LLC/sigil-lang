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

# Output files
STATIC_LIB = libsigil_runtime.a
LTO_LIB = libsigil_runtime_lto.a
TLS_LIB = libsigil_runtime_tls.a
OBJECT = sigil_runtime.o
LTO_OBJECT = sigil_runtime_lto.o
TLS_OBJECT = sigil_runtime_tls.o

.PHONY: all lto tls clean help

all: $(STATIC_LIB)

lto: $(LTO_LIB)

tls: $(TLS_LIB)

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

clean:
	rm -f $(OBJECT) $(LTO_OBJECT) $(TLS_OBJECT) $(STATIC_LIB) $(LTO_LIB) $(TLS_LIB)
