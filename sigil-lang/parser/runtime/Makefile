# Sigil Runtime Library Makefile
#
# Builds the runtime into a static library for faster AOT linking
# Supports LTO for maximum performance

CC ?= clang
AR ?= ar
CFLAGS = -O3 -fPIC -Wall -Wextra

# LTO-enabled flags (works with both gcc and clang)
LTO_CFLAGS = -O3 -fPIC -Wall -Wextra -flto

# Output
STATIC_LIB = libsigil_runtime.a
LTO_LIB = libsigil_runtime_lto.a
OBJECT = sigil_runtime.o
LTO_OBJECT = sigil_runtime_lto.o

.PHONY: all lto clean

all: $(STATIC_LIB)

lto: $(LTO_LIB)

$(OBJECT): sigil_runtime.c
	$(CC) $(CFLAGS) -c sigil_runtime.c -o $(OBJECT)

$(LTO_OBJECT): sigil_runtime.c
	$(CC) $(LTO_CFLAGS) -c sigil_runtime.c -o $(LTO_OBJECT)

$(STATIC_LIB): $(OBJECT)
	$(AR) rcs $(STATIC_LIB) $(OBJECT)
	@echo "Built $(STATIC_LIB)"

$(LTO_LIB): $(LTO_OBJECT)
	gcc-ar rcs $(LTO_LIB) $(LTO_OBJECT) 2>/dev/null || $(AR) rcs $(LTO_LIB) $(LTO_OBJECT)
	@echo "Built $(LTO_LIB) (with LTO)"

clean:
	rm -f $(OBJECT) $(LTO_OBJECT) $(STATIC_LIB) $(LTO_LIB)
