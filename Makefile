CXXFLAGS = -std=c++11 -O3 -Werror -Wall -Wextra -Wshadow
CXX = g++

CPU_EXE = juliaCPU
CPU = Cpu
CL = juliaCL
LINK = lodepng.h lodepng.cpp

$(CPU): $(LINK)
	$(CXX) $(CXXFLAGS) lodepng.cpp $(CPU).cpp -o $(CPU_EXE)
CL:	$(LINK)
	$(CXX) $(CXXFLAGS) lodepng.cpp $(CL).cpp -lOpenCL -o $(CL)
testc:
	./juliaCPU
testcl:
	./juliaCL 64 64
clean:
	@$(RM) -rf *.o *.png *.txt $(CPU_EXE) $(CL)
