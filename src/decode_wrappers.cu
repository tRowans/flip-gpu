#include "decode.cuh"
#include "decode_wrappers.cuh"

//Only used for testing of functions as nvcc doesn't like compiling gtest

void wipeArrayWrap(int N, int* array)
{
    int *d_array;
    cudaMalloc(&d_array, N*sizeof(int));
    cudaMemcpy(d_array, array, N*sizeof(int), cudaMemcpyHostToDevice);
    wipeArray<<<(N+255)/256,256>>>(N, d_array);
    cudaMemcpy(array, d_array, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

void arrayErrorsWrap(int N, unsigned int seed, int* errorTarget, float errorProb)
{
    int *d_errorTarget;
    cudaMalloc(&d_errorTarget, N*sizeof(int));
    cudaMemcpy(d_errorTarget, errorTarget, N*sizeof(int), cudaMemcpyHostToDevice);
    curandState_t *d_states;
    cudaMalloc(&d_states, N*sizeof(curandState_t));
    createStates<<<(N+255)/256,256>>>(N, seed, d_states);
    arrayErrors<<<(N+255)/256,256>>>(N, d_states, d_errorTarget, errorProb);
    cudaMemcpy(errorTarget, d_errorTarget, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_errorTarget);
}

void depolErrorsWrap(int N, unsigned int seed, int* qubitsX, int* qubitsZ, float errorProb)
{
    int *d_qubitsX, *d_qubitsZ;
    cudaMalloc(&d_qubitsX, N*sizeof(int));
    cudaMemcpy(d_qubitsX, qubitsX, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_qubitsZ, N*sizeof(int));
    cudaMemcpy(d_qubitsZ, qubitsZ, N*sizeof(int), cudaMemcpyHostToDevice);
    curandState_t *d_states;
    cudaMalloc(&d_states, N*sizeof(curandState_t));
    createStates<<<(N+255)/256,256>>>(N, seed, d_states);
    depolErrors<<<(N+255)/256,256>>>(N, d_states, d_qubitsX, d_qubitsZ, errorProb);
    cudaMemcpy(qubitsX, d_qubitsX, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(qubitsZ, d_qubitsZ, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qubitsX);
    cudaFree(d_qubitsZ);
}

void flipWrap(int N, int M, int* qubits, int* syndrome, int** bitToChecks, int maxBitDegree)
{
    int *d_qubits, *d_syndrome, *d_bitToChecks;
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, M*sizeof(int));
    cudaMalloc(&d_bitToChecks, maxBitDegree*N*sizeof(int));
    cudaMemcpy(d_qubits, qubits, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bitToChecks, bitToChecks[0], maxBitDegree*N*sizeof(int), cudaMemcpyHostToDevice);
    flip<<<(N+255)/256,256>>>(N, M, d_qubits, d_syndrome, d_bitToChecks, maxBitDegree);
    cudaMemcpy(qubits, d_qubits, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, M*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qubits);
    cudaFree(d_syndrome);
    cudaFree(d_bitToChecks);
}

void pflipWrap(int N, int M, unsigned int seed, int* qubits, int* syndrome, int** bitToChecks, int maxBitDegree)
{
    int *d_qubits, *d_syndrome, *d_bitToChecks;
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, M*sizeof(int));
    cudaMalloc(&d_bitToChecks, maxBitDegree*N*sizeof(int));
    cudaMemcpy(d_qubits, qubits, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bitToChecks, bitToChecks[0], maxBitDegree*N*sizeof(int), cudaMemcpyHostToDevice);
    curandState_t *d_states;
    cudaMalloc(&d_states, N*sizeof(curandState_t));
    createStates<<<(N+255)/256,256>>>(N, seed, d_states);
    pflip<<<(N+255)/256,256>>>(N, M, d_states, d_qubits, d_syndrome, d_bitToChecks, maxBitDegree);
    cudaMemcpy(qubits, d_qubits, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, M*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qubits);
    cudaFree(d_syndrome);
    cudaFree(d_bitToChecks);
}

void calculateSyndromeWrap(int N, int M, int* qubits, int* syndrome, int** checkToBits, int maxCheckDegree)
{
    int *d_qubits, *d_syndrome, *d_checkToBits;
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, M*sizeof(int));
    cudaMalloc(&d_checkToBits, maxCheckDegree*M*sizeof(int));
    cudaMemcpy(d_qubits, qubits, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_checkToBits, checkToBits[0], maxCheckDegree*M*sizeof(int), cudaMemcpyHostToDevice);
    calculateSyndrome<<<(N+255)/256,256>>>(M, d_qubits, d_syndrome, d_checkToBits, maxCheckDegree);
    cudaMemcpy(qubits, d_qubits, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, M*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qubits);
    cudaFree(d_syndrome);
    cudaFree(d_checkToBits);
}
