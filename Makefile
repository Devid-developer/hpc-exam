CC = gcc

CFLAGS = -O3 -Wall -Wextra -march=native -fopenmp -Iinclude -g
ARGS = -x 25000 -y 25000 -n 200 -o 0

SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR ?= build

TEMPLATE_SRC := $(SRC_DIR)/stencil_template_serial.c
FINAL_SRC := $(SRC_DIR)/stencil_serial_final.c

TEMPLATE_O1_BIN := $(BUILD_DIR)/stencil_template_serial_O1
TEMPLATE_O3_BIN := $(BUILD_DIR)/stencil_template_serial_O3
FINAL_O1_BIN := $(BUILD_DIR)/stencil_serial_final_O1
FINAL_O3_BIN := $(BUILD_DIR)/stencil_serial_final_O3
OMP_O3_BIN := $(BUILD_DIR)/stencil_serial_final_omp_O3

# Keep the pure-serial comparison separate from the OpenMP-at-one-thread run.
# CFLAGS contains every requested flag; serial targets remove only -fopenmp.
COMMON_CFLAGS := $(filter-out -O% -fopenmp,$(CFLAGS))
O1_CFLAGS := -O1 $(COMMON_CFLAGS)
O3_CFLAGS := -O3 $(COMMON_CFLAGS)
OMP_O3_CFLAGS := -O3 $(COMMON_CFLAGS) -fopenmp

.PHONY: all template-serial template-o1 template-o3 \
	final-serial final-o1 final-o3 omp-serial \
	run-template-o1 run-template-o3 run-final-o1 run-final-o3 run-omp clean

all: template-serial final-serial omp-serial

template-serial: template-o1 template-o3
template-o1: $(TEMPLATE_O1_BIN)
template-o3: $(TEMPLATE_O3_BIN)

final-serial: final-o1 final-o3
final-o1: $(FINAL_O1_BIN)
final-o3: $(FINAL_O3_BIN)

omp-serial: $(OMP_O3_BIN)

$(BUILD_DIR):
	mkdir -p $@

$(TEMPLATE_O1_BIN): $(TEMPLATE_SRC) $(INCLUDE_DIR)/stencil_template_serial.h | $(BUILD_DIR)
	$(CC) $(O1_CFLAGS) $< -o $@

$(TEMPLATE_O3_BIN): $(TEMPLATE_SRC) $(INCLUDE_DIR)/stencil_template_serial.h | $(BUILD_DIR)
	$(CC) $(O3_CFLAGS) $< -o $@

$(FINAL_O1_BIN): $(FINAL_SRC) $(INCLUDE_DIR)/stencil_serial_final.h | $(BUILD_DIR)
	$(CC) $(O1_CFLAGS) $< -o $@

$(FINAL_O3_BIN): $(FINAL_SRC) $(INCLUDE_DIR)/stencil_serial_final.h | $(BUILD_DIR)
	$(CC) $(O3_CFLAGS) $< -o $@

$(OMP_O3_BIN): $(FINAL_SRC) $(INCLUDE_DIR)/stencil_serial_final.h | $(BUILD_DIR)
	$(CC) $(OMP_O3_CFLAGS) $< -o $@

run-template-o1: $(TEMPLATE_O1_BIN)
	OMP_NUM_THREADS=1 $< $(ARGS)

run-template-o3: $(TEMPLATE_O3_BIN)
	OMP_NUM_THREADS=1 $< $(ARGS)

run-final-o1: $(FINAL_O1_BIN)
	OMP_NUM_THREADS=1 $< $(ARGS)

run-final-o3: $(FINAL_O3_BIN)
	OMP_NUM_THREADS=1 $< $(ARGS)

run-omp: $(OMP_O3_BIN)
	OMP_NUM_THREADS=$${OMP_NUM_THREADS:-32} OMP_PLACES=cores OMP_PROC_BIND=close $< $(ARGS)

clean:
	rm -rf $(BUILD_DIR)
