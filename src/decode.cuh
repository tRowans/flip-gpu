#ifndef DECODE_H
#define DECODE_H

#include<curand.h>
#include<curand_kernel.h>

__global__ void createStates(int N, unsigned int seed, curandState_t* states);

__global__ void wipeArray(int N, int* array);

__global__ void applyErrors(int* lookup, curandState_t* states, int* errorTarget, float errorProb);

__global__ void calculateSyndrome(int* lookup, int* qubits, int* syndrome, int* edgeToFaces);

__global__ void flip(int* qLookup, int* sLookup, int* qubits, int* syndrome, int* faceToEdges, int* edgeToFaces, int* qubitMessages, int* qubitMarginals);

__global__ void pflip(int* qLookup, int* sLookup, int* qubits, int* syndrome, int* faceToEdges, 
                        int* edgeToFaces, int* qubitMessages, int* qubitMarginals, curandState_t* states);

__global__ void updateSyndromeMessages(int* lookup, int* qubitMessages, int* syndrome, int* syndromeMessages, int* edgeToFaces, int* faceToEdges);

__global__ void updateQubitMessages(int* lookup, int* qubitMessages, int* syndromeMessages, int* faceToEdges, int* edgeToFaces, int p);

__global__ void calcMarginals(int* lookup, int* qubitMarginals, int* syndromeMessages, int p);

__global__ void bpCorrection(int* lookup, int* qubits, int* qubitMarginals);

__global__ void measureLogicals(int* lookup, int* qubits, int* nOdd, int L, char bounds);

#endif
