CC ?= gcc
MPICC ?= mpicc

CPPFLAGS ?= -Iinclude
CFLAGS ?= -std=c11 -O3 -Wall -Wextra
OMPFLAGS ?= -fopenmp

OMP_NUM_THREADS ?= 4
MPI_TASKS ?= 4
MPI_LAUNCH ?= srun
RUN_ARGS ?= -x 1000 -y 1000 -n 100 -e 4 -p 1 -F -o 0

SRC_DIR := src
INCLUDE_DIR := include
BUILD_DIR := build

SERIAL_SRC := $(SRC_DIR)/stencil_serial_final.c
PARALLEL_SRC := $(SRC_DIR)/stencil_parallel_final.c

BASE_BIN := $(BUILD_DIR)/stencil_serial_base
SERIAL_BIN := $(BUILD_DIR)/stencil_serial_final
OPENMP_BIN := $(BUILD_DIR)/stencil_serial_final_omp
MPI_BIN := $(BUILD_DIR)/stencil_parallel_final

.PHONY: all base serial openmp omp mpi \
	run-base run-serial run-openmp run-omp run-mpi run clean

all: base serial openmp

base: $(BASE_BIN)

serial: $(SERIAL_BIN)

openmp omp: $(OPENMP_BIN)

mpi: $(MPI_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BASE_BIN): $(SERIAL_SRC) $(INCLUDE_DIR)/stencil_template_serial.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) \
		-DSTENCIL_HEADER='"stencil_template_serial.h"' $< -o $@

$(SERIAL_BIN): $(SERIAL_SRC) $(INCLUDE_DIR)/stencil_serial_final.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@

$(OPENMP_BIN): $(SERIAL_SRC) $(INCLUDE_DIR)/stencil_serial_final.h | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(OMPFLAGS) $< -o $@

$(MPI_BIN): $(PARALLEL_SRC) $(INCLUDE_DIR)/stencil_parallel_final.h | $(BUILD_DIR)
	$(MPICC) $(CPPFLAGS) $(CFLAGS) $< -o $@

run-base: $(BASE_BIN)
	$(BASE_BIN) $(RUN_ARGS)

run-serial run: $(SERIAL_BIN)
	$(SERIAL_BIN) $(RUN_ARGS)

run-openmp run-omp: $(OPENMP_BIN)
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) OMP_PLACES=cores OMP_PROC_BIND=close \
		$(OPENMP_BIN) $(RUN_ARGS)

run-mpi: $(MPI_BIN)
	OMP_NUM_THREADS=1 $(MPI_LAUNCH) --ntasks=$(MPI_TASKS) \
		--cpus-per-task=1 --cpu-bind=cores $(MPI_BIN) $(RUN_ARGS)

clean:
	rm -rf $(BUILD_DIR)
