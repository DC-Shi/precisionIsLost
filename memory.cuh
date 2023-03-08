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



#define cublasErrCheck(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cublas error in %s:%d: %s\n", __FILE__, __LINE__, cublasGetErrorString(status)); \
            exit(1); \
        } \
    } while (0)

const char* cublasGetErrorString(cublasStatus_t status);

// Print the matrix, given the pointer and height, width.
// The data is stored in column-major, but we need to print row-by-row.
// Also, if the matrix too large, we need to print corner values only.
template <typename T>
void printMat(T* P, int uWP, int uHP, int cornerSize = 3)
{
  int i,j;
  for (i = 0; i < uHP; i++)
  {
    if (uHP > 2*cornerSize && cornerSize <= i && i < uHP - cornerSize)
    {
      // Print one line and jump to the final several lines.
      printf("   ... \n");
      i = uHP - cornerSize - 1;
    }
    else
    {
      for (j = 0; j < uWP; j++)
      {
        if (uWP > 2*cornerSize && cornerSize <= j && j < uWP - cornerSize)
        {
            // Print one and jump to the final several columns.
            printf(" ... ");
            j = uWP - cornerSize - 1;
        }
        else
        {
          printf("%8.4f ",P[indexTo1D(i,j,uHP)]);
        }
      }
      printf("\n");
    }
  }
}

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


template <typename T>
__host__ T* initializeDeviceMatFromHostMat(int height, int width, T *hostMatrix)
{
  // Allocate device memory of type float of size height * width called deviceMatrix
  T* deviceMatrix;
  auto cudaStatus = cudaMalloc((void**)&deviceMatrix, height*width*sizeof(T));
  if (cudaStatus != 0) {
    fprintf (stderr, "!!!! device memory allocation error\n");
    return NULL;
  }

  // Set deviceMatrix to values from hostMatrix
  auto status = cublasSetMatrix(height, width, sizeof(T), hostMatrix, height, deviceMatrix, height);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory set error\n");
    return NULL;
  }

  return deviceMatrix;
}


template <typename T>
__host__ T* retrieveDeviceMemory(int height, int width, T *deviceMatrix, T *hostMemory) {
  // TODO get matrix values from deviceMatrix and place results in hostMemory
  auto status = cublasGetMatrix(height, width, sizeof(T), deviceMatrix, height, hostMemory, height);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! host memory get error\n");
    return NULL;
  }

  return hostMemory;
}

__host__ float* initializeFloatFromDouble(int height, int width, double* inputDouble);


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