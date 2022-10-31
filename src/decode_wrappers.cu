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

void applyErrorsWrap(int N, unsigned int seed, int* lookup, int* errorTarget, float errorProb)
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

void flipWrap(int N, int* qLookup, int* sLookup, int* qubits, int* syndrome, int** faceToEdges)
{
    int *d_qLookup, *d_sLookup, *d_qubits, *d_syndrome, *d_faceToEdges;
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, N*sizeof(int));
    cudaMalloc(&d_qLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_sLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_faceToEdges, 4*N*sizeof(int));
    cudaMemcpy(d_qubits, qubits, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qLookup, qLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sLookup, sLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faceToEdges, faceToEdges[0], 4*N*sizeof(int), cudaMemcpyHostToDevice);
    flip<<<(N+255)/256,256>>>(d_qLookup, d_sLookup, d_qubits, d_syndrome, d_faceToEdges);
    cudaMemcpy(qubits, d_qubits, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qLookup);
    cudaFree(d_sLookup);
    cudaFree(d_qubits);
    cudaFree(d_syndrome);
    cudaFree(d_faceToEdges);
}

void pflipWrap(int N, unsigned int seed, int* qLookup, int* sLookup, int* qubits, int* syndrome, int** faceToEdges)
{
    int *d_qLookup, *d_sLookup, *d_qubits, *d_syndrome, *d_faceToEdges;
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, N*sizeof(int));
    cudaMalloc(&d_qLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_sLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_faceToEdges, 4*N*sizeof(int));
    cudaMemcpy(d_qubits, qubits, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qLookup, qLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sLookup, sLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faceToEdges, faceToEdges[0], 4*N*sizeof(int), cudaMemcpyHostToDevice);
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
}

void edgeFlipWrap(int N, unsigned int seed, int* qLookup, int* sLookup, 
                      int* qubits, int* syndrome, int** edgeToFaces, int** faceToEdges)
{
    int *d_qLookup, *d_sLookup, *d_qubits, *d_syndrome, *d_edgeToFaces, *d_faceToEdges;
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, N*sizeof(int));
    cudaMalloc(&d_qLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_sLookup, ((N+255)/256)*256*sizeof(int));
    cudaMalloc(&d_edgeToFaces, 4*N*sizeof(int));
    cudaMalloc(&d_faceToEdges, 4*N*sizeof(int));
    cudaMemcpy(d_qubits, qubits, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qLookup, qLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sLookup, sLookup, ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgeToFaces, edgeToFaces, 4*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_faceToEdges, faceToEdges, 4*N*sizeof(int), cudaMemcpyHostToDevice);
    curandState_t * d_states;
    cudaMalloc(&d_states, N*sizeof(curandState_t));
    createStates<<<(N+255)/256,256>>>(N, seed, d_states);
    edgeFlip<<<(N+255)/256,256>>>(d_qLookup,d_sLookup, d_qubits, d_syndrome, d_edgeToFaces, d_faceToEdges, d_states);
    cudaMemcpy(qubits, d_qubits, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qLookup);
    cudaFree(d_sLookup);
    cudaFree(d_qubits);
    cudaFree(d_syndrome);
    cudaFree(d_edgeToFaces);
    cudaFree(d_faceToEdges);
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
