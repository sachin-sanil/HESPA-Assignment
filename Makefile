CXXFLAGS = -std=c++11 -O3 -Werror -Wall -Wextra -Wshadow
CXX = g++
CUD = nvcc

CPU_EXE = juliaCPU
CPU = Cpu
CL = juliaCL
CUDA = juliaCuda
LINK = lodepng.h lodepng.cpp

$(CPU): $(LINK)
	$(CXX) $(CXXFLAGS) lodepng.cpp $(CPU).cpp -o $(CPU_EXE)
CL:	$(LINK)
	$(CXX) $(CXXFLAGS) lodepng.cpp $(CL).cpp -lOpenCL -o $(CL)
CUDA:   $(LINK)
	$(CUD) -std=c++11 -O3 lodepng.cpp $(CUDA).cu -o $(CUDA)
runc:
	./juliaCPU
runcl:
	./juliaCL 64 64
runcuda:
	./juliaCuda 32 32
clean:
	@$(RM) -rf *.o *.png *.txt $(CPU_EXE) $(CL) $(CUDA)
