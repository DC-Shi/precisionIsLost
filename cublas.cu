#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>

#include "memory.cuh"



int  main (int argc, char** argv) {
  cublasStatus_t status;
  //cublasInit(); // Removed for transisting to cublas v2 API.

  double *A64 = initializeGroundtruthMat<double>(HA, WA, true, -1);
  double *B64 = initializeGroundtruthMat<double>(HB, WB, true, -1);

  // Initialize matrices A and B (2d arrays) based on the HA/WA and HB/WB to be filled with random data
  float *floatA = initializeFloatFromDouble(HA, WA, A64);
  float *floatB = initializeFloatFromDouble(HB, WB, B64);
  // Create arrays of C64 and it should contain the value of A64*B64, as groundtruth.
  double *C64 = initializeGroundtruthMat<double>(HC, WC, true, -1);

  if( A64 == 0 || B64 == 0 || C64 == 0){
    return EXIT_FAILURE;
  } else {
    // Since Ampere support IEEE-compliant FP64 computations, we use cublas for the computation.
    float *floatC = initializeGroundtruthMat<float>(HC, WC, false, 0);
    float *floatCFromCpu = initializeGroundtruthMat<float>(HC, WC, false, -1);
    float alpha = 1.0f;
    float beta = 0.0f;

    // TODO use initializeDeviceMemoryFromHostMemory to create AA from matrix A
    float *devA = initializeDeviceFloatFromHostFloat(HA, WA, floatA);
    // TODO use initializeDeviceMemoryFromHostMemory to create BB from matrix B
    float *devB = initializeDeviceFloatFromHostFloat(HB, WB, floatB);
    // TODO use initializeDeviceMemoryFromHostMemory to create CC from matrix C
    float *devC = initializeDeviceFloatFromHostFloat(HC, WC, floatC);

    cublasHandle_t handle;
    cublasErrCheck( cublasCreate(&handle) );

    // TODO perform Single-Precision Matrix to Matrix Multiplication, GEMM, on AA and BB and place results in CC
    cublasErrCheck(
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HA, WB, WA, &alpha,
          devA, HA,
          devB, HB,
          &beta,
          devC, HC)
    );

    floatC = retrieveDeviceMemory(HC, WC, devC, floatC);

    printf("==== A ====\n");
    printMat(floatA, WA, HA);
    printf("==== B ====\n");
    printMat(floatB, WB, HB);
    printf("==== C ====\n");
    printMat(floatC, WC, HC);

    freeHostPointers(A64, B64, C64, floatA, floatB, floatC);
    freeDevicePointers(devA, devB, devC);
    
    /* Shutdown */
    cublasErrCheck( cublasDestroy(handle) );

    return EXIT_SUCCESS;
  }

}

