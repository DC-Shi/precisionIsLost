#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>

#include "memory.cuh"

// __host__ void printMatrices(float *A, float *B, float *C){
//   printf("\nMatrix A:\n");
//   printMat(A,WA,HA);
//   printf("\n");
//   printf("\nMatrix B:\n");
//   printMat(B,WB,HB);
//   printf("\n");
//   printf("\nMatrix C:\n");
//   printMat(C,WC,HC);
//   printf("\n");
// }

// __host__ int freeMatrices(float *A, float *B, float *C, float *AA, float *BB, float *CC){
//   free( A );  free( B );  free ( C );
//   cublasStatus status = cublasFree(AA);
//   if (status != CUBLAS_STATUS_SUCCESS) {
//     fprintf (stderr, "!!!! memory free error (A)\n");
//     return EXIT_FAILURE;
//   }
//   status = cublasFree(BB);
//   if (status != CUBLAS_STATUS_SUCCESS) {
//     fprintf (stderr, "!!!! memory free error (B)\n");
//     return EXIT_FAILURE;
//   }
//   status = cublasFree(CC);
//   if (status != CUBLAS_STATUS_SUCCESS) {
//     fprintf (stderr, "!!!! memory free error (C)\n");
//     return EXIT_FAILURE;
//   }
//   return EXIT_SUCCESS;
// }

int  main (int argc, char** argv) {
  cublasStatus status;
  cublasInit();

  double *A = initializeGroundtruthMat<double>(HA, WA, true, -1);
  double *B = initializeGroundtruthMat<double>(HB, WB, true, -1);

  // TODO initialize matrices A and B (2d arrays) of floats of size based on the HA/WA and HB/WB to be filled with random data
  float *floatA = initializeFloatFromDouble(HA, WA, A);
  float *floatB = initializeFloatFromDouble(HB, WB, B);

  if( A == 0 || B == 0){
    return EXIT_FAILURE;
  } else {
    // TODO create arrays of floats C filled with random value
    double *C64 = initializeGroundtruthMat<double>(HC, WC, true, -1);
    float *floatC = initializeGroundtruthMat<float>(HC, WC, false, 0);
    float *floatCFromCpu = initializeGroundtruthMat<float>(HC, WC, true, -1);
    float alpha = 1.0f;
    float beta = 0.0f;

    // TODO use initializeDeviceMemoryFromHostMemory to create AA from matrix A
    float *devA = initializeDeviceFloatFromHostFloat(HA, WA, floatA);
    // TODO use initializeDeviceMemoryFromHostMemory to create BB from matrix B
    float *devB = initializeDeviceFloatFromHostFloat(HB, WB, floatB);
    // TODO use initializeDeviceMemoryFromHostMemory to create CC from matrix C
    float *devC = initializeDeviceFloatFromHostFloat(HC, WC, floatC);

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // TODO perform Single-Precision Matrix to Matrix Multiplication, GEMM, on AA and BB and place results in CC
    cublasSgemm('n', 'n', HA, WB, WA, alpha,
          devA, HA,
          devB, HB,
          beta,
          devC, HC);
    status = cublasGetError();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! gemm error (A)\n");
      return EXIT_FAILURE;
    }

    floatC = retrieveDeviceMemory(HC, WC, devC, floatC);

    printf("==== A ====\n");
    printMat(floatA, WA, HA);
    printf("==== B ====\n");
    printMat(floatB, WB, HB);
    printf("==== C ====\n");
    printMat(floatC, WC, HC);
    //printMatrices(floatA, floatB, floatC);

    free(A);
    free(B);
    free(C64);
    free(floatA);
    free(floatB);
    free(floatC);

    cublasFree(devA);
    cublasFree(devB);
    cublasFree(devC);
    
    /* Shutdown */
    status = cublasShutdown();
    if (status != CUBLAS_STATUS_SUCCESS) {
      fprintf (stderr, "!!!! shutdown error (A)\n");
      return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
  }

}

