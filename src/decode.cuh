#ifndef FLIP_H
#define FLIP_H

#include<curand.h>
#include<curand_kernel.h>

#include "code.h"

__global__ void createStates(int N, unsigned int seed, curandState_t* states);

__global__ void wipeArray(int N, int* array);

__global__ void applyErrors(int* lookup, curandState_t* states, int* errorTarget, float errorProb);

__global__ void flip(int* lookup, int* qubits, int* syndrome, int** faceToEdges);

__global__ void pFlip(int* lookup, int* qubits, int* syndrome, int** faceToEdges, curandState_t* states);

__global__ void updateSyndrome(int* lookup, int* qubits, int* syndrome, int** edgeToFaces);

__global__ void measureLogicals(int* lookup, int* qubits, int* nOdd, int L, char bounds)

#endif
