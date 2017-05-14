CXXFLAGS = -std=c++11 -O3 -Werror -Wall -Wextra -Wshadow
CXX = g++

CPU_EXE = juliaCPU
CPU = Cpu
LINK = lodepng.h lodepng.cpp

$(CPU): $(LINK)
	$(CXX) $(CXXFLAGS) lodepng.cpp $(CPU).cpp -o $(CPU_EXE)
testc:
	./juliaCPU

clean:
	@$(RM) -rf *.o *.png $(CPU_EXE) 
