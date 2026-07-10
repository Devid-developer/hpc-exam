CC ?= gcc

CPPFLAGS ?= -Iinclude
CFLAGS ?= -std=c11 -O3 -Wall -Wextra
OMPFLAGS ?= -fopenmp
OMP_NUM_THREADS ?= 4

SRC_DIR := src
BUILD_DIR := build

PROTO_SRC := $(SRC_DIR)/prototipo.c
PROTO_BIN := $(BUILD_DIR)/prototipo
PROTO_OMP_BIN := $(BUILD_DIR)/prototipo_omp

.PHONY: all omp run run-omp clean

all: $(PROTO_BIN)

omp: $(PROTO_OMP_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(PROTO_BIN): $(PROTO_SRC) include/stencil_template_serial.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@

$(PROTO_OMP_BIN): $(PROTO_SRC) include/stencil_template_serial.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) $< -o $@

run: $(PROTO_BIN)
	$(PROTO_BIN) -x 100 -y 100 -n 50 -e 4 -p 0 -F

run-omp: $(PROTO_OMP_BIN)
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) $(PROTO_OMP_BIN) -x 1000 -y 1000 -n 100 -e 4 -p 0 -F

clean:
	rm -rf $(BUILD_DIR)
