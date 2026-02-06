all:
	cd hello && $(MAKE)
	cd matmul && $(MAKE)

clean:
	cd hello && $(MAKE) clean
	cd matmul && $(MAKE) clean

.PHONY : all clean
