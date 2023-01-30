#include "decode.cuh"
#include "decode_wrappers.cuh"

//Only used for testing of functions as nvcc doesn't like compiling gtest

void wipeArrayWrap(int maxIndex, int* array)
{
    int *d_array;
    cudaMalloc(&d_array, maxIndex*sizeof(int));
    cudaMemcpy(d_array, array, maxIndex*sizeof(int), cudaMemcpyHostToDevice);
    wipeArray<<<(N+255)/256,256>>>(maxIndex, d_array);
    cudaMemcpy(array, d_array, maxIndex*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

void arrayErrorsWrap(int maxIndex, unsigned int seed, int* errorTarget, float errorProb)
{
    int *d_errorTarget;
    cudaMalloc(&d_errorTarget, maxIndex*sizeof(int));
    cudaMemcpy(d_errorTarget, errorTarget, maxIndex*sizeof(int), cudaMemcpyHostToDevice);
    curandState_t *d_states;
    cudaMalloc(&d_states, maxIndex*sizeof(curandState_t));
    createStates<<<(maxIndex+255)/256,256>>>(maxIndex, seed, d_states);
    arrayErrors<<<(maxIndex+255)/256,256>>>(maxIndex, d_states, d_errorTarget, errorProb);
    cudaMemcpy(errorTarget, d_errorTarget, maxIndex*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_errorTarget);
}

void depolErrorsWrap(int nQubits, unsigned int seed, int* qubitsX, int* qubitsZ, float errorProb)
{
    int *d_qubitsX, *d_qubitsZ;
    cudaMalloc(&d_qubitsX, nQubits*sizeof(int));
    cudaMemcpy(d_qubitsX, qubitsX, nQubits*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_qubitsZ, nQubits*sizeof(int));
    cudaMemcpy(d_qubitsZ, qubitsZ, nQubits*sizeof(int), cudaMemcpyHostToDevice);
    curandState_t *d_states;
    cudaMalloc(&d_states, nQubits*sizeof(curandState_t));
    createStates<<<(nQubits+255)/256,256>>>(nQubits, seed, d_states);
    depolErrors<<<(nQubits+255)/256,256>>>(nQubits, d_states, d_qubitsX, d_qubitsZ, errorProb);
    cudaMemcpy(qubitsX, d_qubitsX, nQubits*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(qubitsZ, d_qubitsZ, nQubits*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qubitsX);
    cudaFree(d_qubitsZ);
}

void calculateSyndromeWrap(int M, int nQubits, int nChecks, int* qubits, int* syndrome, int** factorToVariables, int* factorDegrees, int maxCheckDegree)
{
    int *d_qubits, *d_syndrome, *d_factorToVariables, *d_factorDegrees;
    cudaMalloc(&d_qubits, nQubits*sizeof(int));
    cudaMalloc(&d_syndrome, M*sizeof(int));
    cudaMalloc(&d_factorToVariables, maxFactorDegree*M*sizeof(int));
    cudaMalloc(&d_factorDegrees, M*sizeof(int));
    cudaMemcpy(d_qubits, qubits, nQubits*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorToVariables, factorToVariables[0], maxFactorDegree*M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorDegrees, factorDegrees, M*sizeof(int), cudaMemcpyHostToDevice);
    calculateSyndrome<<<(nChecks+255)/256,256>>>(nChecks, d_qubits, d_syndrome, d_factorToVariables, d_factorDegrees, maxFactorDegree);
    cudaMemcpy(qubits, d_qubits, nQubits*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, M*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qubits);
    cudaFree(d_syndrome);
    cudaFree(d_factorToVariables);
    cudaFree(d_factorDegrees);
}

void flipWrap(int M, int N, int nQubits, int nChecks, int* qubits, int* syndrome, int** variableToFactors, int* variableDegrees, int maxVariableDegree)
{
    int *d_qubits, *d_syndrome, *d_variableToFactors, *d_variableDegrees;
    cudaMalloc(&d_qubits, nQubits*sizeof(int));
    cudaMalloc(&d_syndrome, M*sizeof(int));
    cudaMalloc(&d_variableToFactors, maxVariableDegree*N*sizeof(int));
    cudaMalloc(&d_variableDegrees, N*sizeof(int));
    cudaMemcpy(d_qubits, qubits, nQubits*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableToFactors, variableToFactors[0], maxVariableDegree*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableDegrees, variableDegrees, N*sizeof(int), cudaMemcpyHostToDevice);
    flip<<<(nQubits+255)/256,256>>>(nQubits, d_qubits, d_syndrome, d_variableToFactors, d_variableDegrees, maxVariableDegree);
    cudaMemcpy(qubits, d_qubits, nQubits*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, M*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qubits);
    cudaFree(d_syndrome);
    cudaFree(d_variableToFactors);
    cudaFree(d_variableDegrees);
}

void pflipWrap(int M, int N, int nQubits, int nChecks, unsigned int seed, int* qubits, int* syndrome, 
        int** variableToFactors, int* variableDegrees, int maxVariableDegree)
{
    int *d_qubits, *d_syndrome, *d_variableToFactors, *d_variableDegrees;
    cudaMalloc(&d_qubits, nQubits*sizeof(int));
    cudaMalloc(&d_syndrome, M*sizeof(int));
    cudaMalloc(&d_variableToFactors, maxVariableDegree*N*sizeof(int));
    cudaMalloc(&d_variableDegrees, N*sizeof(int));
    cudaMemcpy(d_qubits, qubits, nQubits*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableToFactors, variableToFactors[0], maxVariableDegree*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableDegrees, variableDegrees, N*sizeof(int), cudaMemcpyHostToDevice);
    curandState_t *d_states;
    cudaMalloc(&d_states, nQubits*sizeof(curandState_t));
    createStates<<<(nQubits+255)/256,256>>>(nQubits, seed, d_states);
    pflip<<<(nQubits+255)/256,256>>>(nQubits, d_states, d_qubits, d_syndrome, d_variableToFactors, d_variableDegrees, maxVariableDegree);
    cudaMemcpy(qubits, d_qubits, nQubits*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, M*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_qubits);
    cudaFree(d_syndrome);
    cudaFree(d_variableToFactors);
    cudaFree(d_variableDegrees);
}

void initVariableMessagesWrap(int M, int nChecks, double** variableMessages, int* factorDegrees, int maxFactorDegree, double llrp0, double llrq0)
{
    double *d_variableMessages;
    int *d_factorDegrees;
    cudaMalloc(&d_variableMessages, maxFactorDegree*M*sizeof(double));
    cudaMalloc(&d_factorDegrees, M*sizeof(int));
    cudaMemcpy(d_variableMessages, variableMessages[0], maxFactorDegree*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorDegrees, factorDegrees, M*sizeof(double), cudaMemcpyHostToDevice);
    initVariableMessages<<<(M+255)/256,256>>>(M, nChecks, d_variableMessages, d_factorDegrees, maxFactorDegree, llrp0, llrq0);
    cudaMemcpy(variableMessages[0], d_variableMessages, maxFactorDegree*M*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_variableMessages);
    cudaFree(d_factorDegrees);
}

void updateFactorMessagesTanhWrap(int M, int N, double** variableMessages, double** factorMessages, int* syndrome, 
        int** factorToVariables, int* factorDegrees, int maxFactorDegree, int** factorToPos, int maxVariableDegree)
{
    double *d_variableMessages, *d_factorMessages;
    int *d_syndrome, *d_factorToVariables, *d_factorDegrees, *d_factorToPos;
    cudaMalloc(&d_variableMessages, maxFactorDegree*M*sizeof(double));
    cudaMalloc(&d_factorMessages, maxVariableDegree*N*sizeof(double));
    cudaMalloc(&d_syndrome, M*sizeof(int));
    cudaMalloc(&d_factorToVariables, maxFactorDegree*M*sizeof(int));
    cudaMalloc(&d_factorDegrees, M*sizeof(int));
    cudaMalloc(&d_factorToPos, maxFactorDegree*M*sizeof(int));
    cudaMemcpy(d_variableMessages, variableMessages[0], maxFactorDegree*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorMessages, factorMessages[0], maxVariableDegree*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorToVariables, factorToVariables[0], maxFactorDegree*M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorDegrees, factorDegrees, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorToPos, factorToPos[0], maxFactorDegree*M*sizeof(int), cudaMemcpyHostToDevice);
    updateFactorMessagesTanh<<<(M+255)/256,256>>>(M, d_variableMessages, d_factorMessages, d_syndrome, 
            d_factorToVariables, d_factorDegrees, maxFactorDegree, d_factorToPos, maxVariableDegree);
    cudaMemcpy(factorMessages[0], d_factorMessages, maxVariableDegree*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_variableMessages);
    cudaFree(d_factorMessages);
    cudaFree(d_syndrome);
    cudaFree(d_factorToVariables);
    cudaFree(d_factorDegrees);
    cudaFree(d_factorToPos);
}

void updateFactorMessagesMinSum(int alpha, int M, int N, double** variableMessages, double** factorMessages, int* syndrome
        int** factorToVariables, int* factorDegrees, int maxFactorDegree, int** factorToPos, int maxVariableDegree)
{
    double *d_variableMessages, *d_factorMessages;
    int *d_syndrome, *d_factorToVariables, *d_factorDegrees, *d_factorToPos;
    cudaMalloc(&d_variableMessages, maxFactorDegree*M*sizeof(double));
    cudaMalloc(&d_factorMessages, maxVariableDegree*N*sizeof(double));
    cudaMalloc(&d_syndrome, M*sizeof(int));
    cudaMalloc(&d_factorToVariables, maxFactorDegree*M*sizeof(int));
    cudaMalloc(&d_factorDegrees, M*sizeof(int));
    cudaMalloc(&d_factorToPos, maxFactorDegree*M*sizeof(int));
    cudaMemcpy(d_variableMessages, variableMessages[0], maxFactorDegree*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorMessages, factorMessages[0], maxVariableDegree*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorToVariables, factorToVariables[0], maxFactorDegree*M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorDegrees, factorDegrees, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorToPos, factorToPos[0], maxFactorDegree*M*sizeof(int), cudaMemcpyHostToDevice);
    updateFactorMessagesMinSum<<<(M+255)/256,256>>>(alpha, M, d_variableMessages, d_factorMessages, d_syndrome, 
            d_factorToVariables, d_factorDegrees, maxFactorDegree, d_factorToPos, maxVariableDegree);
    cudaMemcpy(factorMessages[0], d_factorMessages, maxVariableDegree*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_variableMessages);
    cudaFree(d_factorMessages);
    cudaFree(d_syndrome);
    cudaFree(d_factorToVariables);
    cudaFree(d_factorDegrees);
    cudaFree(d_factorToPos);
}

void updateVariableMessagesWrap(int M, int N, int nQubits, double** factorMessages, double** variableMessages, int** variableToFactors, 
        int* variableDegrees, int maxVariableDegree, int** variableToPos, int maxFactorDegree, int llrp0, int llrq0)
{
    double *d_factorMessages, *d_variableMessages;
    int *d_variableToFactors, *d_variableDegrees, *d_variableToPos;
    cudaMalloc(&d_factorMessages, maxVariableDegree*N*sizeof(double));
    cudaMalloc(&d_variableMessages, maxFactorDegree*M*sizeof(double));
    cudaMalloc(&d_variableToFactors, maxVariableDegree*N*sizeof(int));
    cudaMalloc(&d_variableDegrees, N*sizeof(int));
    cudaMalloc(&d_variableToPos, maxVariableDegree*N*sizeof(int));
    cudaMemcpy(d_factorMessages, factorMessages[0], maxVariableDegree*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableMessages, variableMessages[0], maxFactorDegree*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableToFactors, variableToFactors[0], maxVariableDegree*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableDegrees, variableDegrees, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableToPos, variableToPos[0], maxVariableDegree*N*sizeof(int), cudaMemcpyHostToDevice);
    updateVariableMessages<<<(N+255)/256,256>>>(N, nQubits, d_factorMessages, d_variableMessages, 
            d_variableToFactors, d_variableDegrees, maxVariableDegree, d_variableToPos, maxFactorDegree, llrp0, llrq0);
    cudaMemcpy(variableDegrees[0], d_variableDegrees, maxFactorDegree*M*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_factorMessages);
    cudaFree(d_variableMessages);
    cudaFree(d_variableToFactors);
    cudaFree(d_variableDegrees);
    cudaFree(d_variableToPos);
}

void calcMarginalsWrap(int N, int nQubits, double* marginals, double** factorMessages, double llrp0, double llrq0, int maxVariableDegree)
{
    double d_marginals, d_factorMessages;
    cudaMalloc(&d_marginals, N*sizeof(double));
    cudaMalloc(&d_factorMessages, maxVariableDegree*N*sizeof(double));
    cudaMemcpy(&d_factorMessages, factorMessages[0], maxVariableDegree*N*sizeof(double), cudaMemcpyHostToDevice);
    calcMarginals<<<(N+255)/256,256>>>(N, nQubits, d_marginals, d_factorMessages, llrp0, llrq0);
    cudaMemcpy(marginals, d_marginals, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_marginals);
    cudaFree(d_factorMessages);
}

void bpCorrectionWrap(int M, int N, int nQubits, int nChecks, double* marginals, 
        int* qubits, int* syndrome, int** variableToFactors, int* variableDegrees, int maxVariableDegree)
{
    double *d_marginals;
    int *d_qubits, *d_syndrome, *d_variableToFactors, *d_variableDegrees;
    cudaMalloc(&d_marginals, N*sizeof(double));
    cudaMalloc(&d_qubits, nQubits*sizeof(int));
    cudaMalloc(&d_syndrome, M*sizeof(int));
    cudaMalloc(&d_variableToFactors, maxVariableDegree*N*sizeof(int));
    cudaMalloc(&d_variableDegrees, N*sizeof(int));
    cudaMemcpy(d_marginals, marginals, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qubits, qubits, nQubits*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_syndrome, syndrome, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableToFactors, variableToFactors[0], maxVariableDegree*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableDegrees, variableDegrees, N*sizeof(int), cudaMemcpyHostToDevice);
    bpCorrection<<<(N+255)/256,256>>>(N, nQubits, nChecks, d_marginals, d_qubits, d_syndrome, 
            d_variableToFactors, d_variableDegrees, maxVariableDegree);
    cudaMemcpy(qubits, d_qubits, nQubits*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(syndrome, d_syndrome, M*sizeof(int), cudaMemcpyDeviceToHost);
}
