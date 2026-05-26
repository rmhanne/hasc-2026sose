all:
	cd hello && $(MAKE)
	cd cppthreads && $(MAKE)
	cd scalar_product && $(MAKE)
	cd matmul && $(MAKE)
	cd nbody && $(MAKE)
	cd benchmarking && $(MAKE)
	cd stream && $(MAKE)
	cd transpose && $(MAKE)
	cd matvec && $(MAKE)
	cd jacobi && $(MAKE)

clean:
	cd hello && $(MAKE) clean
	cd cppthreads && $(MAKE) clean
	cd scalar_product && $(MAKE) clean
	cd matmul && $(MAKE) clean
	cd nbody && $(MAKE) clean
	cd benchmarking && $(MAKE) clean
	cd stream && $(MAKE) clean
	cd transpose && $(MAKE) clean
	cd matvec && $(MAKE) clean
	cd jacobi && $(MAKE) clean

.PHONY : all clean
