#ifndef PRECISIONISLOST_MEMORY
#define PRECISIONISLOST_MEMORY

#include <random>

#define HA 256
#define WA 256
#define WB 256
#define HB WA 
#define WC WB   
#define HC HA  
#define indexTo1D(i,j,ld) (((j)*(ld))+(i))

void printMat(float*P, int uWP, int uHP, int cornerSize=3);

template <typename T>
__host__ T* initializeGroundtruthMat(int height, int width, bool random, T nonRandomValue)
{
  // Allocate host memory of type float of size height * width called hostMatrix
  T *hostMatrix = (T*)malloc(height * width * sizeof(T));

  // Fill hostMatrix with either random data (if random is true) else set each value to nonRandomValue
  if (random)
  {
    T lower_bound = 0, upper_bound = 2;
    // Use C++11 random generator
    std::uniform_real_distribution<T> unif(lower_bound,upper_bound);
    std::default_random_engine re;
    
    for (int i = 0; i < height*width; i++)  hostMatrix[i] = unif(re);
  }
  else
  {
    for (int i = 0; i < height*width; i++)  hostMatrix[i] = nonRandomValue;
  }

  return hostMatrix;
}


__host__ float* initializeFloatFromDouble(int height, int width, double* inputDouble);
__host__ float *initializeDeviceFloatFromHostFloat(int height, int width, float *hostMatrix) ;
__host__ float *retrieveDeviceMemory(int height, int width, float *deviceMatrix, float *hostMemory);


template<typename... Args>
void freeHostPointers (Args && ... inputs)
{
  // Use lambda to free the inputs one by one
  ([&]
  {
      free(inputs);
  } (), ...);
}


template<typename... Args>
void freeDevicePointers (Args && ... inputs)
{
  // Use lambda to free the inputs one by one
  ([&]
  {
      cudaFree(inputs);
  } (), ...);
}

#endif