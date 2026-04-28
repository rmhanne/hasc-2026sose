# include definitions from root directory
include ../make.def

# define targets
GENERIC_TARGETS = transpose
OPENMP_TARGETS = 
ARM_TARGETS = transpose_neon
INTEL_TARGETS = transpose_avx2
SYCL_TARGETS = 

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

transpose: transpose.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SCALARFLAGS) -o $@ $< $(LFLAGS)

transpose_neon: transpose_neon.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(ARMFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

transpose_avx2: transpose_avx2.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(AVX2FLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

debug:
	@echo $(ARCHITECTURE) $(TARGETS)

clean:
	rm -f *.o $(TARGETS)

.PHONY : all clean
