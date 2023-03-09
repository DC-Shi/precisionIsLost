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
    float alpha = 1.0f, beta = 0.0f;
    double alpha64 = 1.0, beta64 = 1.0;
    
    cublasHandle_t handle;
    cublasErrCheck( cublasCreate(&handle) );
    
    double *devA64, *devB64, *devC64;
    devA64 = initializeDeviceMatFromHostMat(HA, WA, A64);
    devB64 = initializeDeviceMatFromHostMat(HB, WB, B64);
    devC64 = initializeDeviceMatFromHostMat(HC, WC, C64);
    // First compute the ground truth  
    cublasErrCheck(
      cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HA, WB, WA, &alpha64,
          devA64, HA,
          devB64, HB,
          &beta64,
          devC64, HC)
    );
    // Copy the value back, C64 now stored the groundtruth.
    retrieveDeviceMemory<double>(HC, WC, devC64, C64);
    // Now we can free the devX64 pointers
    freeDevicePointers(devA64, devB64, devC64);


    // Init device memory from host memory
    float *devA, *devB, *devC;
    devA = initializeDeviceMatFromHostMat(HA, WA, floatA);
    devB = initializeDeviceMatFromHostMat(HB, WB, floatB);
    devC = initializeDeviceMatFromHostMat(HC, WC, floatC);

    
    // Matrix to Matrix Multiplication, GEMM
    cublasErrCheck(
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HA, WB, WA, &alpha,
          devA, HA,
          devB, HB,
          &beta,
          devC, HC)
    );

    retrieveDeviceMemory(HC, WC, devC, floatC);

    float* C32;
    C32 = initializeGroundtruthMat<float>(HC, WC, false, 0);
    for (int i = 0; i < HC; i++)
    {
      for (int j = 0; j < WC; j++)
      {
        for (int k = 0; k < WA; k++)
        {
          C32[indexTo1D(i,j,HC)] += A64[indexTo1D(i,k,HA)] * B64[indexTo1D(k,j,HB)];
        }
      }
    }

    // Show the difference.
    int i,j;
    double maxAbsErr = 0;
    double maxRelErr = 0;
    double meanRelErr = 0;
    double meanAbsErr = 0;
    for (i = 0; i < HC; i++)
    {
      for (j = 0; j < WC; j++)
      {
        double diff = std::abs(C32[indexTo1D(i,j,HC)] - floatC[indexTo1D(i,j,HC)]);
        maxAbsErr = std::max(diff, maxAbsErr);
        double relDiff = (C32[indexTo1D(i,j,HC)] == 0 ? 0 : std::abs(diff / C32[indexTo1D(i,j,HC)]));
        maxRelErr = std::max(relDiff, maxRelErr);
        meanAbsErr += diff;
        meanRelErr += relDiff;
      }
    }

    printf("The diffenence of float_using_gpu vs FP32_cpu \n");
    printf("Max Abs diff: %e, Max Rel diff: %e\n", maxAbsErr, maxRelErr);
    printf("Avg Abs diff: %e, Avg Rel diff: %e\n", meanAbsErr/(1.0*HC*WC), meanRelErr/(1.0*HC*WC));


    // printf("==== A ====\n");
    // printMat(floatA, WA, HA);
    // printf("==== B ====\n");
    // printMat(floatB, WB, HB);
    printf("==== C64 ====\n");
    printMat(C64, WC, HC);
    printf("==== C32 ====\n");
    printMat(C32, WC, HC);
    printf("==== C ====\n");
    printMat(floatC, WC, HC);

    freeHostPointers(A64, B64, C64, floatA, floatB, floatC, C32);
    freeDevicePointers(devA, devB, devC);
    
    /* Shutdown */
    cublasErrCheck( cublasDestroy(handle) );

    return EXIT_SUCCESS;
  }

}

