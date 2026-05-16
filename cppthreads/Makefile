# include definitions from root directory
include ../make.def

# define targets
GENERIC_TARGETS = peterson barrier philosophers producer_consumer packaged_task power_method
OPENMP_TARGETS = 
ARM_TARGETS = 
INTEL_TARGETS = 
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

peterson: peterson.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking -O0 -o $@ $< $(LFLAGS)

barrier: barrier.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking -O3 -o $@ $< $(LFLAGS)

philosophers: philosophers.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking -O3 -o $@ $< $(LFLAGS)

producer_consumer: producer_consumer.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking -O3 -o $@ $< $(LFLAGS)

packaged_task: packaged_task.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking -O3 -o $@ $< $(LFLAGS)

power_method: power_method.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking -O3 -o $@ $< $(LFLAGS)

debug:
	@echo $(ARCHITECTURE) $(TARGETS)

clean:
	rm -f *.o $(TARGETS)

.PHONY : all clean
