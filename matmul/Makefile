# include definitions from root directory
include ../make.def

ROOTPATH = ../

all: matmul_seq matmul_omp_avx512

matmul_seq: matmul_seq.cc Makefile ../make.def
	$(CC) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(SCALARFLAGS) -o $@ $< $(LFLAGS)

matmul_omp_avx512: matmul_omp_avx512.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(AVX512FLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

matmul_omp_stdsimd: matmul_omp_stdsimd.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(ARMFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

matmul_armneon: matmul_armneon.cc Makefile ../make.def
	$(CCOMP) -I$(ROOTPATH) -I$(ROOTPATH)benchmarking $(CCFLAGSBASE) $(OMPFLAGS) $(ARMFLAGS) -o $@ $< $(LFLAGS) $(LFLAGSOMP)

clean:
	rm -f *.o \
	matmul_seq \
	matmul_omp_avx512 \
	matmul_armneon
