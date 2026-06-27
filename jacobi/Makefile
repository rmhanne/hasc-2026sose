# include definitions from root directory
include ../make.def

# define targets
GENERIC_TARGETS = jacobi_seq
OPENMP_TARGETS = jacobi_omp_stdsimd jacobi_omp gauss_seidel_omp
TBB_TARGETS = jacobi_tbb
ARM_TARGETS = 
INTEL_TARGETS = 
SYCL_TARGETS =
MPI_TARGETS = jacobi_mpi

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
ifdef HAVE_TBB
	TARGETS += $(TBB_TARGETS)
endif
ifdef HAVE_MPI
	TARGETS += $(MPI_TARGETS)
endif

# include path for all targets
ROOTPATH = ../

# the default target
all: $(TARGETS)

jacobi_seq: jacobi_seq.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking -O3 $(SCALARFLAGS) -o $@ $< $(LFLAGS)

jacobi_omp_stdsimd: jacobi_omp_stdsimd.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(STDSIMDFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

jacobi_omp: jacobi_omp.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(VECTORIZATIONFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

gauss_seidel_omp: gauss_seidel_omp.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(VECTORIZATIONFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

jacobi_tbb: jacobi_tbb.cc Makefile ../make.def
	$(CCTBB) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(VECTORIZATIONFLAGS) $(TBBFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSTBB)

jacobi_mpi: jacobi_mpi.cc Makefile ../make.def
	$(CCMPI) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(VECTORIZATIONFLAGS) $(MPIFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSMPI)

debug:
	@echo $(ARCHITECTURE) $(TARGETS)

clean:
	rm -f *.o $(TARGETS)

.PHONY : all clean
