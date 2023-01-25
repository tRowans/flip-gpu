#ifndef DECODE_H
#define DECODE_H

#include<curand.h>
#include<curand_kernel.h>

__global__ void createStates(int N, unsigned int seed, curandState_t* states);

__global__ void wipeArray(int N, int* array);

__global__ void arrayErrors(int maxIndex, curandState_t* states, int* errorTarget, float errorProb);

__global__ void depolErrors(int N, curandState_t* states, int* qubitsX, int* qubitsZ, float errorProb);

__global__ void flip(int N, int* qubits, int* syndrome, int* bitToChecks, int maxBitDegree);

__global__ void pflip(int N, curandState_t* states, int* qubits, int* syndrome, int* bitToChecks, int maxBitDegree);

__global__ void calculateSyndrome(int M, int* qubits, int* syndrome, int* checkToBits, int maxCheckDegree);

#endif
