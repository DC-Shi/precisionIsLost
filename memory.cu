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


const char* cublasGetErrorString(cublasStatus_t status)
{
    switch(status)
    {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE"; 
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH"; 
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED"; 
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR"; 
    }
    return "unknown error";
}
