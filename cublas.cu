#include <stdlib.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <iostream>

#include "memory.cuh"


static void show_usage(std::string name)
{
    std::cerr << "Usage: " << name << " <option(s)> SOURCES\n"
              << "Options:\n"
              << "\t-h,--help\t\tShow this help message\n"
              << "\t-l,--loop LOOP\t\tSpecify the loop times for invoking cublas"
              << std::endl;
}

int  main (int argc, char** argv) {

    std::vector <std::string> sources;
    int numLoop = 5;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            show_usage(argv[0]);
            return 0;
        } else if ((arg == "-l") || (arg == "--loop")) {
            if (i + 1 < argc) { // Make sure we aren't at the end of argv!
                numLoop = std::stoi(argv[++i]); // Increment 'i' so we don't get the argument as the next argv[i].
            } else { // Uh-oh, there was no argument to the loop option.
                  std::cerr << "--loop option requires one argument." << std::endl;
                return 1;
            }  
        } else {
            sources.push_back(argv[i]);
        }
    }

  cublasStatus_t status;
  //cublasInit(); // Removed for transisting to cublas v2 API.

  double *A64 = initializeGroundtruthMat<double>(HA, WA, true, -1);
  double *B64 = initializeGroundtruthMat<double>(HB, WB, true, -1);

  // Initialize matrices A and B (2d arrays) based on the HA/WA and HB/WB to be filled with random data
  float *floatA = initializeFloatFromDouble(HA, WA, A64);
  float *floatB = initializeFloatFromDouble(HB, WB, B64);
  // Create arrays of C64 and it should contain the value of A64*B64, as groundtruth.
  // Must init to zero to give the correct result.
  double *C64 = initializeGroundtruthMat<double>(HC, WC, false, 0);

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

    printf("Will do %d sGEMM\n", numLoop);

    // Create timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // And start timing before GEMM
    cudaEventRecord(start);
    
    for (int i = 0; i < numLoop; i++)
    {
    // Matrix to Matrix Multiplication, GEMM
      cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, HA, WB, WA, &alpha,
          devA, HA,
          devB, HB,
          &beta,
          devC, HC);
    }
    // And end timing after GEMM
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float gemmMilliseconds = 0;
    cudaEventElapsedTime(&gemmMilliseconds, start, stop);
    printf("GEMM takes %.1f ms\n", gemmMilliseconds);

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
        double diff = std::abs(C64[indexTo1D(i,j,HC)] - floatC[indexTo1D(i,j,HC)]);
        maxAbsErr = std::max(diff, maxAbsErr);
        double relDiff = (C64[indexTo1D(i,j,HC)] == 0 ? 0 : std::abs(diff / C64[indexTo1D(i,j,HC)]));
        maxRelErr = std::max(relDiff, maxRelErr);
        meanAbsErr += diff;
        meanRelErr += relDiff;
      }
    }

    printf("The diffenence of float_using_gpu vs FP64 \n");
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

