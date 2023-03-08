#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include "memory.cuh"

// Print the matrix, given the pointer and height, width.
// The data is stored in column-major, but we need to print row-by-row.
// Also, if the matrix too large, we need to print corner values only.
void printMat(float*P, int uWP, int uHP, int cornerSize)
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


__host__ float* initializeFloatFromDouble(int height, int width, double* inputDouble)
{
  float *ret = (float*)malloc(height * width * sizeof(float));

  for (int i = 0; i < height*width; i++)  ret[i] = inputDouble[i];

  return ret;
}

__host__ float *initializeDeviceFloatFromHostFloat(int height, int width, float *hostMatrix)
{
  // Allocate device memory of type float of size height * width called deviceMatrix
  float* deviceMatrix;
  auto cudaStatus = cudaMalloc((void**)&deviceMatrix, height*width*sizeof(float));
  if (cudaStatus != 0) {
    fprintf (stderr, "!!!! device memory allocation error\n");
    return NULL;
  }

  // Set deviceMatrix to values from hostMatrix
  auto status = cublasSetMatrix(height, width, sizeof(float), hostMatrix, height, deviceMatrix, height);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory set error\n");
    return NULL;
  }

  return deviceMatrix;
}

__host__ float *retrieveDeviceMemory(int height, int width, float *deviceMatrix, float *hostMemory) {
  // TODO get matrix values from deviceMatrix and place results in hostMemory
  auto status = cublasGetMatrix(height, width, sizeof(float), deviceMatrix, height, hostMemory, height);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! host memory get error\n");
    return NULL;
  }

  return hostMemory;
}


__host__ void printMatrices(float *A, float *B, float *C){
  printf("\nMatrix A:\n");
  printMat(A,WA,HA);
  printf("\n");
  printf("\nMatrix B:\n");
  printMat(B,WB,HB);
  printf("\n");
  printf("\nMatrix C:\n");
  printMat(C,WC,HC);
  printf("\n");
}


__host__ int freeMatrices(float *A, float *B, float *C, float *AA, float *BB, float *CC){
  free( A );  free( B );  free ( C );
  cublasStatus status = cublasFree(AA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasFree(BB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }
  status = cublasFree(CC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}


