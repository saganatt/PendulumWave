all:
	nvcc -O3 -std=c++11 -Xptxas="-v" -gencode arch=compute_35,code=compute_35 -gencode arch=compute_35,code=sm_35  ./src/kernel.cu -o des-cracker
