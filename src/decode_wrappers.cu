#include "decode.cuh"
#include "decode_wrappers.cuh"
#include<iostream>

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

void applyErrorsWrap(int N, unsigned int seed, int* lookup, int* errorTarget, double errorProb)
{
    int *d_lookup, *d_errorTarget;
    cudaMalloc(&d_lookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_errorTarget, N*sizeof(int));
    cudaMemcpy(d_lookup, lookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_errorTarget, errorTarget, N*sizeof(int), cudaMemcpyHostToDevice);
    curandState_t *d_states;
    cudaMalloc(&d_states, N*sizeof(curandState_t));
    createStates<<<(N+255)/256,256>>>(N, seed, d_states);
    applyErrors<<<(N+255)/256,256>>>(d_lookup, d_states, d_errorTarget, errorProb);
    cudaMemcpy(errorTarget, d_errorTarget, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_lookup);
    cudaFree(d_errorTarget);
}

void calculateSyndromeWrap(int N, int* lookup, int* qubits, int* syndrome, int** edgeToFaces)
{
    int *d_lookup, *d_qubits, *d_syndrome, *d_edgeToFaces;
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, N*sizeof(int));
    cudaMalloc(&d_lookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_edgeToFaces, 4*N*sizeof(int));
    cudaMemcpy(d_qubits, qubits, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lookup, lookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeToFaces, edgeToFaces[0], 4*N*sizeof(int), cudaMemcpyHostToDevice);
    calculateSyndrome<<<(N+255)/256,256>>>(d_lookup, d_qubits, d_syndrome, d_edgeToFaces);
    cudaMemcpy(qubits, d_qubits, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_lookup);
    cudaFree(d_qubits);
    cudaFree(d_syndrome);
    cudaFree(d_edgeToFaces);
}

void flipWrap(int N, int* qLookup, int* sLookup, int* qubits, int* syndrome, int** faceToEdges)
{
    int *d_qLookup, *d_sLookup, *d_qubits, *d_syndrome, *d_faceToEdges, *d_edgeToFaces;
    double *d_variableMessages, *d_qubitMarginals;
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, N*sizeof(int));
    cudaMalloc(&d_qLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_sLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_faceToEdges, 4*N*sizeof(int));
    cudaMalloc(&d_edgeToFaces, 4*N*sizeof(int));
    cudaMemcpy(d_qubits, qubits, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qLookup, qLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sLookup, sLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faceToEdges, faceToEdges[0], 4*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeToFaces, edgeToFaces[0], 4*N*sizeof(int), cudaMemcpyHostToDevice);
    flip<<<(N+255)/256,256>>>(d_qLookup, d_sLookup, d_qubits, d_syndrome, d_faceToEdges);
    cudaMemcpy(qubits, d_qubits, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qLookup);
    cudaFree(d_sLookup);
    cudaFree(d_qubits);
    cudaFree(d_syndrome);
    cudaFree(d_faceToEdges);
    cudaFree(d_edgeToFaces);
}

void pflipWrap(int N, unsigned int seed, int* qLookup, int* sLookup, int* qubits, int* syndrome, int** faceToEdges)
{
    int *d_qLookup, *d_sLookup, *d_qubits, *d_syndrome, *d_faceToEdges, *d_edgeToFaces;
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, N*sizeof(int));
    cudaMalloc(&d_qLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_sLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_faceToEdges, 4*N*sizeof(int));
    cudaMalloc(&d_edgeToFaces, 4*N*sizeof(int));
    cudaMemcpy(d_qubits, qubits, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qLookup, qLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sLookup, sLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faceToEdges, faceToEdges[0], 4*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeToFaces, edgeToFaces[0], 4*N*sizeof(int), cudaMemcpyHostToDevice);
    curandState_t *d_states;
    cudaMalloc(&d_states, N*sizeof(curandState_t));
    createStates<<<(N+255)/256,256>>>(N, seed, d_states);
    pflip<<<(N+255)/256,256>>>(d_qLookup, d_sLookup, d_qubits, d_syndrome, d_faceToEdges, d_states);
    cudaMemcpy(qubits, d_qubits, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qLookup);
    cudaFree(d_sLookup);
    cudaFree(d_qubits);
    cudaFree(d_syndrome);
    cudaFree(d_faceToEdges);
    cudaFree(d_edgeToFaces);
}

void initVariableMessagesWrap(int N, int* lookup, double* variableMessages, double llr0, double llrq0)
{
    int *d_sLookup;
    double *d_variableMessages;
    cudaMalloc(&d_sLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_variableMessages, 8*N*sizeof(double));
    cudaMemcpy(d_sLookup, lookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    initVariableMessages<<<(N+255)/256,256>>>(d_sLookup, d_variableMessages, llr0, llrq0);
    cudaMemcpy(variableMessages, d_variableMessages, 5*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sLookup);
    cudaFree(d_variableMessages);
}

void updateFactorMessagesWrap(int N, int* lookup, double* variableMessages, int* syndrome, double* factorMessages, int** edgeToFaces, int** faceToEdges)
{
    int *d_sLookup, *d_syndrome, *d_edgeToFaces, *d_faceToEdges;
    double *d_variableMessages, *d_factorMessages;
    cudaMalloc(&d_sLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_variableMessages, 8*N*sizeof(double));
    cudaMalloc(&d_syndrome, N*sizeof(int));
    cudaMalloc(&d_factorMessages, 5*N*sizeof(double));
    cudaMalloc(&d_edgeToFaces, 4*N*sizeof(int));
    cudaMalloc(&d_faceToEdges, 4*N*sizeof(int));
    cudaMemcpy(d_sLookup, lookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableMessages, variableMessages, 5*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeToFaces, edgeToFaces[0], 4*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faceToEdges, faceToEdges[0], 4*N*sizeof(int), cudaMemcpyHostToDevice);
    updateFactorMessages<<<(N+255)/256,256>>>(d_sLookup, d_variableMessages, d_syndrome, d_factorMessages, d_edgeToFaces, d_faceToEdges, N);
    cudaMemcpy(factorMessages, d_factorMessages, 5*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_sLookup);
    cudaFree(d_variableMessages);
    cudaFree(d_syndrome);
    cudaFree(d_factorMessages);
    cudaFree(d_edgeToFaces);
    cudaFree(d_faceToEdges);
}

void updateVariableMessagesWrap(int N, int* lookup, double* variableMessages, double* factorMessages, int** faceToEdges, int** edgeToFaces, double llr0)
{
    int *d_qLookup, *d_faceToEdges, *d_edgeToFaces;
    double *d_variableMessages, *d_factorMessages;
    cudaMalloc(&d_qLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_variableMessages, 5*N*sizeof(double));
    cudaMalloc(&d_factorMessages, 5*N*sizeof(double));
    cudaMalloc(&d_faceToEdges, 4*N*sizeof(int));
    cudaMalloc(&d_edgeToFaces, 4*N*sizeof(int));
    cudaMemcpy(d_qLookup, lookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorMessages, factorMessages, 5*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faceToEdges, faceToEdges[0], 4*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeToFaces, edgeToFaces[0], 4*N*sizeof(int), cudaMemcpyHostToDevice);
    updateVariableMessages<<<(N+255)/256,256>>>(d_qLookup, d_variableMessages, d_factorMessages, d_faceToEdges, d_edgeToFaces, llr0);
    cudaMemcpy(variableMessages, d_variableMessages, 5*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_qLookup);
    cudaFree(d_variableMessages);
    cudaFree(d_factorMessages);
    cudaFree(d_faceToEdges);
    cudaFree(d_edgeToFaces);
}

void calcMarginalsWrap(int N, int* qLookup, int* sLookup, double* qubitMarginals, double* stabMarginals, double* factorMessages, double llr0, double llrq0)
{
    int *d_qLookup, *d_sLookup;
    double *d_qubitMarginals, *d_stabMarginals, *d_factorMessages;
    cudaMalloc(&d_qLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_sLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_qubitMarginals, N*sizeof(double));
    cudaMalloc(&d_stabMarginals, N*sizeof(double));
    cudaMalloc(&d_factorMessages, 5*N*sizeof(double));
    cudaMemcpy(d_qLookup, qLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sLookup, sLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qubitMarginals, qubitMarginals, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stabMarginals, stabMarginals, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorMessages, factorMessages, 5*N*sizeof(double), cudaMemcpyHostToDevice);
    calcMarginals<<<(N+255)/256,256>>>(d_qLookup, d_sLookup, d_qubitMarginals, d_stabMarginals, d_factorMessages, llr0, llrq0, N);
    cudaMemcpy(qubitMarginals, d_qubitMarginals, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(stabMarginals, d_stabMarginals, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_qLookup);
    cudaFree(d_sLookup);
    cudaFree(d_qubitMarginals);
    cudaFree(d_stabMarginals);
    cudaFree(d_factorMessages);
}

void bpCorrectionWrap(int N, int* qLookup, int* sLookup, int* qubits, double* qubitMarginals, int* syndrome, double* stabMarginals, int* faceToEdges);
{
    int *d_qLookup, *d_sLookup, *d_qubits, *d_syndrome, *d_faceToEdges;
    double *d_qubitMarginals, *d_stabMarginals;
    cudaMalloc(&d_qLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_sLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, N*sizeof(int));
    cudaMalloc(&d_faceToEdges, 4*N*sizeof(int));
    cudaMalloc(&d_qubitMarginals, N*sizeof(double));
    cudaMalloc(&d_stabMarginals, N*sizeof(double));
    cudaMemcpy(d_qLookup, qLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sLookup, sLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qubits, qubits, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faceToEdges, faceToEdges[0], 4*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qubitMarginals, qubitMarginals, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_stabMarginals, stabMarginals, N*sizeof(double), cudaMemcpyHostToDevice);
    bpCorrection<<<(N+255)/256,256>>>(d_qLookup, d_sLookup, d_qubits, d_qubitMarginals, d_syndrome, d_stabMarginals, d_faceToEdges);
    cudaMemcpy(qubits, d_qubits, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qLookup);
    cudaFree(d_sLookup);
    cudaFree(d_qubits);
    cudaFree(d_qubitMarginals);
    cudaFree(d_syndrome);
    cudaFree(d_stabMarginals);
    cudaFree(d_faceToEdges);
}

void measureLogicalsWrap(int N, int* lookup, int* qubits, int &nOdd, int L, char bounds)
{
    int *d_lookup, *d_qubits, *d_nOdd;
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_lookup, ((3*L*L+63)/64)*64*sizeof(int));
    cudaMemcpy(d_qubits, qubits, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lookup, lookup, ((3*L*L+63)/64)*64*sizeof(int), cudaMemcpyHostToDevice);
    cudaMallocManaged(&d_nOdd, sizeof(int));
    *d_nOdd = 0;
    measureLogicals<<<(3*L*L+63)/64,64>>>(d_lookup, d_qubits, d_nOdd, L, bounds);
    cudaDeviceSynchronize();
    nOdd = *d_nOdd;
    cudaFree(d_lookup);
    cudaFree(d_qubits);
    cudaFree(d_nOdd);
}
