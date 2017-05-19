#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void color_init(__global unsigned char* colourbit,const int img_size,const double h,const int max_iter)
{
			uint idx = get_global_id(0); //x value
			uint idy = get_global_id(1); //y value
            double c_real = -0.8;
            double c_img = 0.2;
            //int img_size = img_size1;
            //double h = 4.0/(double)img_size;
            //int max_iter = 50;
            
            double rel = 0.0; double img = 0.0;
            rel = (double)idx*h-2.0;
            img = (double)idy*h-2.0;
            double temp = 0.0;
            double mod2 = 0.0;
            mod2 = rel*rel + img*img;
            int iter = 0;
            while(iter < max_iter && mod2 < 100)
            {
                temp = rel*rel - img*img + c_real;
                img = 2*rel*img +c_img;
                rel = temp;
                mod2 = rel*rel + img*img;
                ++iter;
            }
            double inv = 1/(double)max_iter;
            
            colourbit[4 * img_size*idy + 4*idx + 3] = 255;
			colourbit[4 * img_size*idy + 4*idx + 2] = 0;                       //((double)num*0.02)*255;//(num >> 8)%255;
			colourbit[4 * img_size*idy + 4*idx + 1] = 0;
			colourbit[4 * img_size*idy + 4*idx + 0] = ((double)iter*inv)*255;
            
            //itr_x[idx] = ((double)iter*inv)*255;
			//itr_y[idy] = ((double)iter*inv)*255;
}
//__global double* itr_x, __global double* itr_y
