# include definitions from root directory
include ../make.def

# define targets
GENERIC_TARGETS = matmul_seq
OPENMP_TARGETS = matmul_omp_stdsimd
ARM_TARGETS = matmul_omp_armneon
INTEL_TARGETS = matmul_omp_avx512
SYCL_TARGETS = matmul_sycl

TARGETS = $(GENERIC_TARGETS)
ifdef HAVE_OPENMP
	TARGETS += $(OPENMP_TARGETS)
endif
ifeq ($(ARCHITECTURE),intel)
	TARGETS += $(INTEL_TARGETS)
endif
ifeq ($(ARCHITECTURE),arm)
	TARGETS += $(ARM_TARGETS)
endif
ifdef HAVE_SYCL
	TARGETS += $(SYCL_TARGETS)
endif

# include path for all targets
ROOTPATH = ../

# the default target
all: $(TARGETS)

matmul_seq: matmul_seq.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SCALARFLAGS) -o $@ $< $(LFLAGS)

matmul_omp_stdsimd: matmul_omp_stdsimd.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(STDSIMDFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

matmul_omp_avx512: matmul_omp_avx512.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(AVX512FLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

matmul_omp_armneon: matmul_omp_armneon.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(ARMFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

matmul_sycl: matmul_sycl.cc Makefile ../make.def
	$(DPCPP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(DPCPPFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSDPCPP)

debug:
	@echo $(ARCHITECTURE) $(TARGETS)

clean:
	rm -f *.o $(TARGETS)

.PHONY : all clean
