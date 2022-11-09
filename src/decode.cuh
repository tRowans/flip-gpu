#ifndef DECODE_H
#define DECODE_H

#include<curand.h>
#include<curand_kernel.h>

__global__ void createStates(int N, unsigned int seed, curandState_t* states);

__global__ void wipeArray(int N, int* array);

__global__ void applyErrors(int* lookup, curandState_t* states, int* errorTarget, float errorProb);

__global__ void calculateSyndrome(int* lookup, int* qubits, int* syndrome, int* edgeToFaces);

__global__ void flip(int* qLookup, int* sLookup, int* qubits, int* syndrome, int* faceToEdges, int* edgeToFaces, float* qubitMessages, float* qubitMarginals);

__global__ void pflip(int* qLookup, int* sLookup, int* qubits, int* syndrome, int* faceToEdges, 
                        int* edgeToFaces, float* qubitMessages, float* qubitMarginals, curandState_t* states);

__global__ void updateSyndromeMessages(int* lookup, float* qubitMessages, int* syndrome, float* syndromeMessages, int* edgeToFaces, int* faceToEdges);

__global__ void updateQubitMessages(int* lookup, float* qubitMessages, float* syndromeMessages, int* faceToEdges, int* edgeToFaces, float p);

__global__ void calcMarginals(int* lookup, float* qubitMarginals, float* syndromeMessages, float p);

__global__ void bpCorrection(int* lookup, int* qubits, float* qubitMarginals);

__global__ void measureLogicals(int* lookup, int* qubits, int* nOdd, int L, char bounds);

#endif
