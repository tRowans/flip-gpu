#ifndef DECODE_H
#define DECODE_H

#include<curand.h>
#include<curand_kernel.h>

__global__ void createStates(int N, unsigned int seed, curandState_t* states);

__global__ void wipeArray(int N, int* array);

__global__ void applyErrors(int* lookup, curandState_t* states, int* errorTarget, double errorProb);

__global__ void calculateSyndrome(int* lookup, int* qubits, int* syndrome, int* edgeToFaces);

__global__ void flip(int* qLookup, int* sLookup, int* qubits, int* syndrome, int* faceToEdges, int* edgeToFaces, double* variableMessages, double* qubitMarginals);

__global__ void pflip(int* qLookup, int* sLookup, int* qubits, int* syndrome, int* faceToEdges, 
                        int* edgeToFaces, double* variableMessages, double* qubitMarginals, curandState_t* states);

__global__ void initVariableMessages(int* lookup, double* variableMessages, double llr0, double llrq0);

__global__ void updateFactorMessages(int* lookup, double* variableMessages, int* syndrome, double* factorMessages, int* edgeToFaces, int* faceToEdges, int N);

__global__ void updateVariableMessages(int* lookup, double* variableMessages, double* factorMessages, int* faceToEdges, int* edgeToFaces, double llr0);

__global__ void calcMarginals(int* qLookup, int* sLookup, double* qubitMarginals, double* stabMarginals, double* factorMessages, double llr0, double llrq0, int N);

__global__ void calcQubitMarginals(int* lookup, double* qubitMarginals, double* factorMessages, double llr0);

__global__ void calcStabMarginals(int* lookup, double* stabMarginals, double* factorMessages, double llrq0, int N);

__global__ void bpSyndromeCorrection(int* lookup, int* syndrome, double* factorMessages, double* stabMarginals, int* edgeToFaces, int* faceToEdges, int N);

__global__ void bpCorrection(int* lookup, int* qubits, double* marginals);

__global__ void measureLogicals(int* lookup, int* qubits, int* nOdd, int L, char bounds);

#endif
