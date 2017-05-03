#include <cuda_runtime.h>

#include <cstddef>
#include <sys/time.h>
#include <iostream>

#include <vector>


void checkError( cudaError_t err)
{
	if(err != cudaSuccess)

	{
		std::cout << cudaGetErrorString(err) << std::endl;
		exit(-1);
	}
}

	//global is a kernel: global cannot be called from host, but can be called from functions
	__global__ void kernel(int*A, int* B, int* C, long long N)
{
	// amount of threads per block and number of blocks. in OpenCL we also specify total number of threads
	long long idx = blockIdx.x * blockDim.x+ threadIdx.x; //all these indexes start at zero;
	//cuda launching kernels is possible in 2D/3D as well;
	if(idx < N)
	C[idx] = A[idx];
	
} 

double getSeconds()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double) tp.tv_sec + (double) tp.tv_usec * 1e-6);
}

int main()
{
	const long long nElem = 1 << 24;
	//std::cout << nElem << std::endl;
	std::vector<int> A(nElem, 1);
	std::vector<int> B(nElem, 1);
	std::vector<int> C(nElem, 0);
	double start, end;
	const long long nBytes = nElem * sizeof(int);
	std::cout << nBytes * 1e-6 << std::endl;

	/*allocate the memory to allocate memory on GPU
	** cudaMalloc() malloc call: returns pointer to the first block of memeory.
	** no difference between gpu and cpu mem pointer.
	** nomenclature for gpu pounter: d_variable 
	*/
	int* d_A;
	int* d_B;
	int* d_C;	
	checkError(cudaMalloc(&d_A, nBytes));
	checkError(cudaMalloc(&d_B, nBytes));
	checkError(cudaMalloc(&d_C, nBytes));

	checkError(cudaMemcpy(d_A, &A[0], nBytes, cudaMemcpyHostToDevice)); //  A/B/C are local memory
	checkError(cudaMemcpy(d_B, &B[0], nBytes, cudaMemcpyHostToDevice));
	
	start = getSeconds();
	kernel<<< (1 << 14), (1 << 10) >>> (d_A, d_B, d_C, nElem); //number of blocks
	checkError( cudaPeekAtLastError() );
	checkError(cudaDeviceSynchronize());	
	end = getSeconds();
	std::cout << "time is " << end - start << std::endl; // this actually gives the time needed to launch the kernel. to get the correct measurements, we have to synchronize. this ensures that all kernels have finished before it exectutes the next instruction.
	// checkError should be used for every cuda intruction. 
	checkError(cudaMemcpy(&C[0], d_C, nBytes, cudaMemcpyDeviceToHost));  //destination:source:size:direction
	for (auto c : C)
	{
		if (c!=1)
		{
			std::cout << "error" << std::endl;
			exit(123);
		}
	}
	checkError(cudaFree(d_A));
	checkError(cudaFree(d_B));
	checkError(cudaFree(d_C));
}

//cuda: nvidia compiler: compiles the kernel
//()nvcc  -{to be safe}std=c++11 -{compute-capability}arch=sm_20 source.cu && ./a.out
