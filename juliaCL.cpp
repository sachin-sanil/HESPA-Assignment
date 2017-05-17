#include<iostream>
#include<CL/cl.hpp>
#include <sstream>
#include <memory>
#include<fstream>
#include "lodepng.h"
#include<sys/time.h>

//Encode from raw pixels to disk with a single function call
//he image argument has width * height RGBA pixels or width * height * 4 bytes
void encodeImage(const char* filename, std::vector<unsigned char>& image, unsigned width, unsigned height)
{
	//Encode the image
	unsigned error = lodepng::encode(filename, image, width, height);

	//if there's an error, display it
	if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
}

double getSeconds()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int main(int argc, char* argv[])
{
	std::stringstream s;
	if(argc>1){
    s << argv[1];
    }
    else
    {
        std::cout<< "Number of threads/work items not given " <<std::endl;
        return 0;
    }
	int work_items_per_group=0;
	s >> work_items_per_group; // threads per block/ work items per work group
	int img_size = 2048; // image pixle size
	//int work_groups = (img_size*img_size) / work_items_per_group; //total number of work groups
	double spacing = 4.0 / (double) img_size;
	double h = spacing;
	int max_iterations = 200;
	//std::vector<unsigned char> color_bit_host(img_size * 4, 55);
	std::vector <unsigned char> colourbit;
	colourbit.resize(img_size*img_size * 4);
	
	//get platform
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	//choose a platform
	std::cout << "Platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;

	//choose device
	std::vector<cl::Device> devices;
	platforms[0].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
	std::cout << "Device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;

	//create context
	cl::Context context(devices);

	// Create a command queue
	cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

	//create memory buffer on devices
	cl::Buffer colourBit_device = cl::Buffer(context, CL_MEM_READ_WRITE, img_size *img_size * 4* sizeof(unsigned char));
	cl::Buffer buffer_h = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double));
	cl::Buffer max_iter_dev = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
	cl::Buffer img_size_dev = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
	
	// Copy the input data to the input buffers using the command queue.
	queue.enqueueWriteBuffer(buffer_h, CL_FALSE, 0, sizeof(double), &h);
	queue.enqueueWriteBuffer(max_iter_dev, CL_FALSE, 0, sizeof(int), &max_iterations);
	queue.enqueueWriteBuffer(img_size_dev, CL_FALSE, 0, sizeof(int), &img_size);
	
	// Read the program source
	std::ifstream sourceFile("kernel_color.cl");
	std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
	cl::Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()));
	
	// Make program from the source code
	cl::Program program = cl::Program(context, source);
	
	// Build the program for the devices
	if (program.build(devices) != CL_SUCCESS) {
		std::cout << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0]) << "\n";
		exit(1);
	}
	
	// Make kernel
	cl::Kernel color_kernel(program, "color_init");
	
	// Set the kernel arguments
	color_kernel.setArg(0, colourBit_device);
	color_kernel.setArg(1, &buffer_h);
	color_kernel.setArg(2, &max_iter_dev);
	color_kernel.setArg(3, &img_size_dev);
    
    double wcTimestart =0.0;
    double wcTimeend = 0.0;
    wcTimestart = getSeconds();
	//execute kernel
	cl::NDRange global(img_size, img_size);
	cl::NDRange local(work_items_per_group, work_items_per_group);
	queue.enqueueNDRangeKernel(color_kernel, cl::NullRange, global, local);
	
	//copy colourbit from device to host
	queue.enqueueReadBuffer(colourBit_device, CL_TRUE, 0, img_size *img_size * 4 * sizeof(unsigned char), &colourbit[0]);
    wcTimeend =getSeconds();
	//encode image
	encodeImage("JuliaCPU.png", colourbit, img_size, img_size);
    std::cout << "Time taken for kernel execution and writeback: " << (wcTimeend-wcTimestart)*1e3 << "milli-sec" <<std::endl;
	std::cout << "The image has been generated and is named as JuliaCPU.png" << std::endl;	
	return 0;
}
