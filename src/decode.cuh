#ifndef DECODE_H
#define DECODE_H

#include<curand.h>
#include<curand_kernel.h>

#include "code.h"

__global__ void createStates(int N, unsigned int seed, curandState_t* states);

__global__ void wipeArrays(int N, int* qubits, int* syndrome);

__global__ void applyErrors(int* lookup, curandState_t* states, int* errorTarget, float errorProb);

__global__ void flip(int* lookup, int* qubits, int* syndrome, int* faceToEdges);

__global__ void pflip(int* lookup, int* qubits, int* syndrome, int* faceToEdges, curandState_t* states);

__global__ void updateSyndrome(int* lookup, int* qubits, int* syndrome, int* edgeToFaces);

__global__ void measureLogicals(int* lookup, int* qubits, int* nOdd, int L, char bounds);

#endif
