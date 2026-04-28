# include definitions from root directory
include ../make.def

# define targets
GENERIC_TARGETS = matvec
OPENMP_TARGETS = 
ARM_TARGETS = 
INTEL_TARGETS = matvec_avx2
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

matvec: matvec.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SCALARFLAGS) -o $@ $< $(LFLAGS)

matvec_avx2: matvec_avx2.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(AVX2FLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

debug:
	@echo $(ARCHITECTURE) $(TARGETS)

clean:
	rm -f *.o $(TARGETS)

.PHONY : all clean
