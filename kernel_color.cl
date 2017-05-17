#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable
__kernel void color_init(global unsigned char* colourBit, global double* h, global int* maxiterations,global int* img_size)
{
			const int idx = get_global_id(0); //x value
			const int idy = get_global_id(1); //y value
            double c_real = -0.8;
			double c_img = 0.2;
			double mod = 0.0;
			double mod2 = 0.0;
			double real=0.0, img= 0.0;
            //printf("%f\n",*h);
			//initialise the pixels
            double h1 = 4/(2048*2048);
			real = (double)idx*h[0];
			img  = (double)idy*h[0];
			mod2 = real*real + img*img;
			mod = sqrt(mod2);
			double temp=0.0;
			int iter = 0;
			while (mod<=10 && iter< maxiterations[0])
			{
				temp = real*real - img*img + c_real;
				img = 2*real*img + c_img;
				real = temp;
				mod2 = real*real + img*img;
				mod = sqrt(mod2);
				++iter;
			}
			double inv = 1.0/(double)maxiterations[0];
			//update colour
			colourBit[4 *(img_size[0])*idy + 4 * idx + 3] = 255;
			colourBit[4 *(img_size[0])*idy + 4 * idx + 2] = 0;
			colourBit[4 *(img_size[0])*idy + 4 * idx + 1] = 0;
			colourBit[4 *(img_size[0])*idy + 4 * idx + 0] = ((double)iter*inv)*255;
}
