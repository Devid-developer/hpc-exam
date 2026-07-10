CC ?= gcc

CPPFLAGS ?= -Iinclude
CFLAGS ?= -std=c11 -O3 -Wall -Wextra
OMPFLAGS ?= -fopenmp
OMP_NUM_THREADS ?= 4

SRC_DIR := src
BUILD_DIR := build

SERIAL_SRC := $(SRC_DIR)/stencil_serial_final.c
SERIAL_BIN := $(BUILD_DIR)/stencil_serial_final
SERIAL_OMP_BIN := $(BUILD_DIR)/stencil_serial_final_omp

.PHONY: all omp run run-omp clean

all: $(SERIAL_BIN)

omp: $(SERIAL_OMP_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(SERIAL_BIN): $(SERIAL_SRC) include/stencil_serial_final.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@

$(SERIAL_OMP_BIN): $(SERIAL_SRC) include/stencil_serial_final.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) $< -o $@

run: $(SERIAL_BIN)
	$(SERIAL_BIN) -x 100 -y 100 -n 50 -e 4 -p 0 -F

run-omp: $(SERIAL_OMP_BIN)
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) $(SERIAL_OMP_BIN) -x 1000 -y 1000 -n 100 -e 4 -p 0 -F

clean:
	rm -rf $(BUILD_DIR)
