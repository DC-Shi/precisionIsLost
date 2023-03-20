# IDIR=./
CXX = nvcc

CXXFLAGS += $(shell pkg-config --cflags --libs opencv4)
LDFLAGS += $(shell pkg-config --libs --static opencv)

all: clean build

build: cublas.cu memory.cu
	$(CXX) $? --std c++17 -o cublas_assignment.exe -Wno-deprecated-gpu-targets $(CXXFLAGS) -I. -I/usr/local/cuda/include -lcuda -lcublas

run: build
	./cublas_assignment.exe

clean:
	rm -f cublas_assignment.exe output.txt 
