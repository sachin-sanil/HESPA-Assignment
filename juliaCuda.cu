#include <cuda_runtime.h>
#include <sys/time.h>
#include <iostream>
#include<stdio.h>
#include <string>
#include "lodepng.h"


//Encode from raw pixels to disk with a single function call
//he image argument has width * height RGBA pixels or width * height * 4 bytes

typedef double type_T;
void encodeImage(const char* filename, unsigned char const* image, unsigned width, unsigned height)
{
	//Encode the image
	unsigned error = lodepng::encode(filename, image, width, height);
	//if there's an error, display it
	if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}

void checkError (cudaError_t err)
{
    if(err != cudaSuccess )
    {
        std::cout<< cudaGetErrorString(err) <<std::endl ;
        exit(-1);
    }
}

double getSeconds()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

void printDevProp(cudaDeviceProp devProp)
{
	    std::cout<<"\t+++++++++++++++++++++++++++++++++DeviceProperty+++++++++++++++++++++++++++++++++"<<std::endl;
        std::cout<<"\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
        std::cout<<"\tMajor revision number:         "<<  devProp.major<<std::endl;
        std::cout<<"\tMinor revision number:         "<< devProp.minor<<std::endl;
		std::cout<<"\tName:                          "<< devProp.name<<std::endl;
		std::cout<<"\tTotal global memory:           "<< devProp.totalGlobalMem<<std::endl;
		std::cout<<"\tTotal shared memory per block: "<< devProp.sharedMemPerBlock<<std::endl;
		std::cout<<"\tTotal registers per block:     "<<  devProp.regsPerBlock<<std::endl;
        std::cout<<"\tWarp size:                     "<<devProp.warpSize<<std::endl;
        std::cout<<"\tMaximum memory pitch:          "<<devProp.memPitch<<std::endl;
        std::cout<<"\tMaximum threads per block:     "<<devProp.maxThreadsPerBlock<<std::endl;
            for (int i = 0; i < 3; ++i)
            {
        std::cout<<"\tMaximum dimension %d of block:  "<<i<<devProp.maxThreadsDim[i]<<std::endl;}
            for (int i = 0; i < 3; ++i){
        std::cout<<"\tMaximum dimension %d of grid:   "<<i<<devProp.maxGridSize[i]<<std::endl;}
        std::cout<<"\tClock rate:                     "<<devProp.clockRate<<std::endl;
        std::cout<<"\tTotal constant memory:          "<<devProp.totalConstMem<<std::endl;
        std::cout<<"\tTexture alignment:              "<<devProp.textureAlignment<<std::endl;
        std::cout<<"\tConcurrent copy and execution:  "<<(devProp.deviceOverlap ? "Yes" : "No")<<std::endl;
        std::cout<<"\tNumber of multiprocessors:      "<<devProp.multiProcessorCount<<std::endl;
        std::cout<<"\tKernel execution timeout:       "<<(devProp.kernelExecTimeoutEnabled ? "Yes" : "No")<<std::endl;
        std::cout<<"\t+++++++++++++++++++++++++++++++++DeviceProperty+++++++++++++++++++++++++++++++++"<<std::endl;
        std::cout<<"\t++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"<<std::endl;
		//return;
}

__global__ void evalJulia(type_T h,

			  unsigned int max_iteration,

			  type_T pixel_limit,

			  type_T c_real,

			  type_T c_img,

			  unsigned char* colourBit,
              
			  long img_size){

	
	int id = threadIdx.x + blockIdx.x*blockDim.x;
    long x_index = id%img_size;
	long y_index = id/img_size;
    type_T real = -2.0 +( h * (x_index));
	type_T img = -2.0 + (h * (y_index));
	type_T mod = real*real + img*img;
	type_T temp=0;
	int iter = 0;
	
	while ((mod <= (pixel_limit*pixel_limit)) && (iter < max_iteration))
		{
		
		temp = (real*real) - (img*img) + c_real;
		img = 2.0*real*img + c_img;
		real = temp;
		mod = (real * real) + (img * img);
		
		iter = iter + 1;
		}
	
    colourBit[4 *(img_size)*y_index + 4 * x_index + 3] = 255;
    colourBit[4 *(img_size)*y_index + 4 * x_index + 2] = 0;
    colourBit[4 *(img_size)*y_index + 4 * x_index + 1] = (iter/200.0)*255;
    colourBit[4 *(img_size)*y_index + 4 * x_index + 0] = 0;
                            
}




int main(int argc , char *argv[])
{
	unsigned int threadX = std::atol(argv[1]);
    unsigned int threadY = std::atol(argv[2]);
    
    int devCount;
    cudaGetDeviceCount(&devCount);
    std::cout<<"There are "<<devCount<< "CUDA devices"<<std::endl; 
    for (int ic = 0; ic < devCount; ++ic)
    {
	       // Get device properties
		        std::cout<<"nCUDA Device #"<< ic <<std::endl;
                cudaDeviceProp devProp;
				cudaGetDeviceProperties(&devProp, ic);
                printDevProp(devProp);
    }
    long img_size = 2048; // Image size(64x64)
		
	type_T spacing = 4.0 / (type_T)img_size; //spacing is length/image size
	type_T pixel_limit = 20;
	type_T c_real = 0.0;	//constant complex real
	type_T c_img = 0.8; 	//constant complex imaginary
	unsigned int iteration_limit = 200; // maximum number of iterations
	unsigned char*  colourBit = new unsigned char[img_size*img_size * 4];
	unsigned char* d_colourBit;
	cudaMalloc((void**)&d_colourBit, (img_size*img_size*4*sizeof(unsigned char)));
	double wcTimeStart= 0.0, wcTimeEnd=0.0;
    //computuation begins here:
	wcTimeStart = getSeconds(); //Start time
	long threads_Block = threadX*threadY;
	long blocks = (img_size*img_size)/threads_Block;
	evalJulia<<<blocks, threads_Block>>>(spacing, iteration_limit, pixel_limit, c_real, c_img, d_colourBit, img_size);
    cudaDeviceSynchronize();
	cudaError_t errSync  = cudaGetLastError();
	cudaError_t errAsync = cudaDeviceSynchronize();
	if (errSync != cudaSuccess) 
 		 printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
	if (errAsync != cudaSuccess)
  	printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
	cudaMemcpy(colourBit,d_colourBit, (4*img_size*img_size*sizeof(unsigned char)),cudaMemcpyDeviceToHost);
	wcTimeEnd = getSeconds();
	std::cout << "Done with operations, begin image encoding!" << std::endl;
	std::cout << "Time Taken for computation: " << wcTimeEnd-wcTimeStart << " sec" << std::endl;
	encodeImage("cudaJuliaCPU.png", colourBit, img_size, img_size);
	std::cout << "The image has been generated and is named as JuliaCPU.png" << std::endl;
	std::cout << "Time Taken for image encoding: " << (wcTimeEnd-wcTimeStart)*1e3 << " milli-sec" << std::endl;
	cudaFree(d_colourBit);
	delete(colourBit);
	return 0;
}


