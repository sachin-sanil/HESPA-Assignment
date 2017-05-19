#include<iostream>
#include<CL/cl.hpp>
#include <sstream>
#include <memory>
#include<fstream>
#include "lodepng.h"
#include<sys/time.h>


double getSeconds()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}
void get_device_info();
int main(int argc, char* argv[])
{
	std::stringstream s;
	if(argc>1){
    s << argv[1] << " " << argv[2];
    }
    else
    {
      std::cout<< "Number of threads/work items not given " <<std::endl;
        return 0;
    }
	int x_work_items_per_group=0, y_work_items_per_group= 0;
	s >> x_work_items_per_group >> y_work_items_per_group; // threads per block/ work items per work groups
	//std::cout<< x_work_items_per_group << " " << y_work_items_per_group <<std::endl;
	
	const int img_size = 2048; // image pixle size
	int x_work_groups = img_size/x_work_items_per_group; //work groups containing work_items_per_group threads/work items
	int y_work_groups = img_size/y_work_items_per_group;
	const double spacing = 4.0/(double)img_size;
    const int max_iteration = 200;
	    
	unsigned char* colourbit = new unsigned char[img_size*img_size*4]; //colorbit to store pixel info. 
    	
	//get platform
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	//choose a platform
	std::cout << "Platform: " << platforms[0].getInfo<CL_PLATFORM_NAME>() << std::endl;
    //char buffer[1024];
	//choose device
	std::vector<cl::Device> devices;
	platforms[0].getDevices(CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_CPU, &devices);
	
    get_device_info();
    /*std::cout << "Device: " << devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
    std::cout << "Vendor: " << devices[0].getInfo<CL_DEVICE_VENDOR>() << std::endl;
    std::cout << "Device Version: " << devices[0].getInfo<CL_DEVICE_VERSION >() << std::endl;
    std::cout << "Driver Version: " << devices[0].getInfo<CL_DRIVER_VERSION>() << std::endl;
    std::cout << "Device_OpenCL_C_VERSION: " << devices[0].getInfo<CL_DEVICE_OPENCL_C_VERSION>() << std::endl;
    std::cout << "Vendor: " << devices[0].getInfo<CL_DEVICE_VENDOR>() << std::endl;
    std::cout << "Extension: " << devices[0].getInfo<CL_DEVICE_EXTENSIONS>()<< std::endl;*/
    
	//create context
	cl::Context context(devices);

	// Create a command queue
	cl::CommandQueue queue = cl::CommandQueue(context, devices[0]);

	//create memory buffer on devices
	cl::Buffer colourBit_device = cl::Buffer(context, CL_MEM_READ_WRITE, img_size *img_size * 4* sizeof(unsigned char));
	
    //cl::Buffer h_dev = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(double));
	//cl::Buffer max_iter_dev = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
	//cl::Buffer img_size_dev = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    
	//cl::Buffer iteration_xd = cl::Buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(double));
	//cl::Buffer iteration_yd = cl::Buffer(context, CL_MEM_READ_WRITE, img_size * sizeof(double));

	//double* iteration_x = new double[img_size];
	//double* iteration_y = new double[img_size];
	// Copy the input data to the input buffers using the command queue.
	
	//queue.enqueueWriteBuffer(*h_dev, CL_TRUE, 0, sizeof(double), &spacing);
	//queue.enqueueWriteBuffer(*max_iter_dev, CL_TRUE, 0, sizeof(int), &max_iterations);
	//queue.enqueueWriteBuffer(*img_size_dev, CL_TRUE, 0, sizeof(int), &img_size);
	
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
	color_kernel.setArg(1, img_size);
    color_kernel.setArg(2, spacing);
    color_kernel.setArg(3, max_iteration);
	    
    double wcTimestart =0.0;
    double wcTimeend = 0.0;
    wcTimestart = getSeconds();
	//execute kernel
	cl::NDRange global(img_size, img_size);
	cl::NDRange local(x_work_groups, y_work_groups);
	queue.enqueueNDRangeKernel(color_kernel, cl::NullRange, global, local);
	//copy colourbit from device to host
	queue.enqueueReadBuffer(colourBit_device, CL_TRUE, 0, img_size * img_size * 4 * sizeof(unsigned char), colourbit);
    wcTimeend = getSeconds();
	//queue.enqueueReadBuffer(iteration_xd, CL_TRUE ,0, img_size*sizeof(double), iteration_x);
	//queue.enqueueReadBuffer(iteration_yd, CL_TRUE ,0, img_size*sizeof(double), iteration_y);
    std::cout << "Number of threads in x and y :" << x_work_items_per_group << " " << y_work_items_per_group << std::endl;
    std::cout << "The time to execute pixel calc. using OpenCL for max "<< max_iteration << " iterations: " << (wcTimeend-wcTimestart)*1e3<< " milli-sec"<< std::endl;
	//encode image
    lodepng::encode("JuliaCL.png", colourbit, img_size, img_size);

	std::cout << "The image has been generated and is named as JuliaCL.png" << std::endl;
	//std::fstream fl;
	/*fl.open("iter.txt",std::fstream::in |std::fstream::out|std::fstream::app);
	for(int i =0; i < img_size; ++i)
		fl<< iteration_x[i] <<" " << iteration_y[i] << std::endl;
	fl.close();*/	

	delete(colourbit);
	//delete(iteration_x);
	//delete(iteration_y);
	return 0;
}

void get_device_info(){
    
    char buffer[1024];
	cl_uint num_platforms;
	clGetPlatformIDs(0, NULL, &num_platforms);
	cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
	clGetPlatformIDs(num_platforms, platforms, NULL);
     std::cout<< "****************************************"<< std::endl;  
    cl_platform_id platform = platforms[0];
		clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(buffer), buffer, NULL);
		printf("PLATFORM_NAME: %s\n", buffer);
		clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(buffer), buffer, NULL);
		printf("PLATFORM_VENDOR: %s\n", buffer);
		clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(buffer), buffer, NULL);
		printf("PLATFORM_VERSION: %s\n", buffer);
		clGetPlatformInfo(platform, CL_PLATFORM_PROFILE, sizeof(buffer), buffer, NULL);
		printf("PLATFORM_PROFILE: %s\n", buffer);
		clGetPlatformInfo(platform, CL_PLATFORM_EXTENSIONS, sizeof(buffer), buffer, NULL);
		printf("PLATFORM_EXTENSIONS: %s\n", buffer);
		printf("\n");
    std::cout<< "****************************************"<< std::endl;   
      cl_uint num_devices;
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
      cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        cl_device_id device = devices[0];
        clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, sizeof(buffer), buffer, NULL);
        printf("DEVICE_EXTENSIONS: %s\n", buffer);
        cl_ulong global_mem_size;
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_mem_size), &global_mem_size, NULL);
        printf("DEVICE_GLOBAL_MEM_SIZE: %lu B = %lu MB\n", global_mem_size, global_mem_size / 1048576);
        cl_ulong local_mem_size;
        clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem_size), &local_mem_size, NULL);
        printf("DEVICE_LOCAL_MEM_SIZE: %lu B = %lu KB\n", local_mem_size, local_mem_size / 1024);
        cl_uint max_clock_frequency;
        clGetDeviceInfo(device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock_frequency), &max_clock_frequency, NULL);
        printf("DEVICE_MAX_CLOCK_FREQUENCY: %u MHz\n", max_clock_frequency);
        cl_uint max_compute_units;
        clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units), &max_compute_units, NULL);
        printf("DEVICE_MAX_COMPUTE_UNITS: %u\n", max_compute_units);
        size_t max_work_group_size;
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);
        printf("DEVICE_MAX_WORK_GROUP_SIZE: %lu\n", max_work_group_size);
        cl_uint max_work_item_dimensions;
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(max_work_item_dimensions), &max_work_item_dimensions, NULL);
		printf("DEVICE_MAX_WORK_ITEM_DIMENSIONS: %u\n", max_work_item_dimensions);
        size_t* max_work_item_sizes = (size_t*)malloc(sizeof(size_t) * max_work_item_dimensions);
        clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * max_work_item_dimensions, max_work_item_sizes, NULL);
		printf("DEVICE_MAX_WORK_ITEM_SIZES: "); for (size_t i = 0; i < max_work_item_dimensions; ++i) printf("%lu\t", max_work_item_sizes[i]); printf("\n");
		free(max_work_item_sizes);
        clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
        printf("DEVICE_NAME: %s\n", buffer);
		clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(buffer), buffer, NULL);
		printf("DEVICE_VENDOR: %s\n", buffer);
		cl_uint vendor_id;
		clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID, sizeof(vendor_id), &vendor_id, NULL);
		printf("DEVICE_VENDOR_ID: %d\n", vendor_id);
		clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
		printf("DEVICE_VERSION: %s\n", buffer);
		clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(buffer), buffer, NULL);
		printf("DRIVER_VERSION: %s\n", buffer);
		clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(buffer), buffer, NULL);
		printf("DEVICE_OPENCL_C_VERSION: %s\n", buffer);
        std::cout<< "****************************************"<< std::endl;   
}
