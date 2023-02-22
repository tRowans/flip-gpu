#include "decode.cuh"
#include "decode_wrappers.cuh"

//Only used for testing of functions as nvcc doesn't like compiling gtest

void wipeArrayWrap(int maxIndex, int* array)
{
    int *d_array;
    cudaMalloc(&d_array, maxIndex*sizeof(int));
    cudaMemcpy(d_array, array, maxIndex*sizeof(int), cudaMemcpyHostToDevice);
    wipeArray<<<(maxIndex+255)/256,256>>>(maxIndex, d_array);
    cudaMemcpy(array, d_array, maxIndex*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_array);
}

void depolErrorsWrap(int N_X, int N_Z, int nQubits, unsigned int seed, int* variablesX, int* variablesZ, float errorProb)
{
    int *d_variablesX, *d_variablesZ;
    cudaMalloc(&d_variablesX, N_X*sizeof(int));
    cudaMemcpy(d_variablesX, variablesX, N_X*sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&d_variablesZ, N_Z*sizeof(int));
    cudaMemcpy(d_variablesZ, variablesZ, N_Z*sizeof(int), cudaMemcpyHostToDevice);
    curandState_t *d_states;
    if (N_X > N_Z)
    {
        cudaMalloc(&d_states, N_X*sizeof(curandState_t));
        createStates<<<(N_X+255)/256,256>>>(N_X, seed, d_states);
    }
    else
    {
        cudaMalloc(&d_states, N_Z*sizeof(curandState_t));
        createStates<<<(N_Z+255)/256,256>>>(N_X, seed, d_states);
    }
    depolErrors<<<(nQubits+255)/256,256>>>(nQubits, d_states, d_variablesX, d_variablesZ, errorProb);
    cudaMemcpy(variablesX, d_variablesX, N_X*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(variablesZ, d_variablesZ, N_Z*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_variablesX);
    cudaFree(d_variablesZ);
}

void measErrorsWrap(int nQubits, int nChecks, unsigned int seed, int* variables, float errorProb)
{
    int *d_variables;
    int N = nQubits + nChecks;
    cudaMalloc(&d_variables, N*sizeof(int));
    cudaMemcpy(d_variables, variables, N*sizeof(int), cudaMemcpyHostToDevice);
    curandState_t *d_states;
    cudaMalloc(&d_states, N*sizeof(curandState_t));
    createStates<<<(N+255)/256,256>>>(N, seed, d_states);
    measErrors<<<(N+255)/256,256>>>(nQubits, nChecks, d_states, d_variables, errorProb);
    cudaMemcpy(variables, d_variables, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_variables);
}

void calculateSyndromeWrap(int N, int M, int* variables, int* factors, int** factorToVariables, int* factorDegrees, int maxFactorDegree)
{
    int *d_variables, *d_factors, *d_factorToVariables, *d_factorDegrees;
    cudaMalloc(&d_variables, N*sizeof(int));
    cudaMalloc(&d_factors, M*sizeof(int));
    cudaMalloc(&d_factorToVariables, maxFactorDegree*M*sizeof(int));
    cudaMalloc(&d_factorDegrees, M*sizeof(int));
    cudaMemcpy(d_variables, variables, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factors, factors, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorToVariables, factorToVariables[0], maxFactorDegree*M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorDegrees, factorDegrees, M*sizeof(int), cudaMemcpyHostToDevice);
    calculateSyndrome<<<(M+255)/256,256>>>(M, d_variables, d_factors, d_factorToVariables, d_factorDegrees, maxFactorDegree);
    cudaMemcpy(variables, d_variables, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(factors, d_factors, M*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_variables);
    cudaFree(d_factors);
    cudaFree(d_factorToVariables);
    cudaFree(d_factorDegrees);
}

void flipWrap(int N, int M, int nQubits, int nChecks, int* variables, int* factors, int** variableToFactors, int* variableDegrees, int maxVariableDegree)
{
    int *d_variables, *d_factors, *d_variableToFactors, *d_variableDegrees;
    cudaMalloc(&d_variables, N*sizeof(int));
    cudaMalloc(&d_factors, M*sizeof(int));
    cudaMalloc(&d_variableToFactors, maxVariableDegree*N*sizeof(int));
    cudaMalloc(&d_variableDegrees, N*sizeof(int));
    cudaMemcpy(d_variables, variables, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factors, factors, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableToFactors, variableToFactors[0], maxVariableDegree*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableDegrees, variableDegrees, N*sizeof(int), cudaMemcpyHostToDevice);
    flip<<<(nQubits+255)/256,256>>>(nQubits, d_variables, d_factors, d_variableToFactors, d_variableDegrees, maxVariableDegree);
    cudaMemcpy(variables, d_variables, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(factors, d_factors, M*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_variables);
    cudaFree(d_factors);
    cudaFree(d_variableToFactors);
    cudaFree(d_variableDegrees);
}

void pflipWrap(int N, int M, int nQubits, int nChecks, unsigned int seed, int* variables, int* factors, 
        int** variableToFactors, int* variableDegrees, int maxVariableDegree)
{
    int *d_variables, *d_factors, *d_variableToFactors, *d_variableDegrees;
    cudaMalloc(&d_variables, N*sizeof(int));
    cudaMalloc(&d_factors, M*sizeof(int));
    cudaMalloc(&d_variableToFactors, maxVariableDegree*N*sizeof(int));
    cudaMalloc(&d_variableDegrees, N*sizeof(int));
    cudaMemcpy(d_variables, variables, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factors, factors, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableToFactors, variableToFactors[0], maxVariableDegree*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableDegrees, variableDegrees, N*sizeof(int), cudaMemcpyHostToDevice);
    curandState_t *d_states;
    cudaMalloc(&d_states, N*sizeof(curandState_t));
    createStates<<<(N+255)/256,256>>>(N, seed, d_states);
    pflip<<<(nQubits+255)/256,256>>>(nQubits, d_states, d_variables, d_factors, d_variableToFactors, d_variableDegrees, maxVariableDegree);
    cudaMemcpy(variables, d_variables, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(factors, d_factors, M*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_variables);
    cudaFree(d_factors);
    cudaFree(d_variableToFactors);
    cudaFree(d_variableDegrees);
}

void initVariableMessagesWrap(int M, int nChecks, double* variableMessages, int* factorDegrees, int maxFactorDegree, double llrp0, double llrq0)
{
    double *d_variableMessages;
    int *d_factorDegrees;
    cudaMalloc(&d_variableMessages, maxFactorDegree*M*sizeof(double));
    cudaMalloc(&d_factorDegrees, M*sizeof(int));
    cudaMemcpy(d_variableMessages, variableMessages, maxFactorDegree*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorDegrees, factorDegrees, M*sizeof(double), cudaMemcpyHostToDevice);
    initVariableMessages<<<(M+255)/256,256>>>(M, nChecks, d_variableMessages, d_factorDegrees, maxFactorDegree, llrp0, llrq0);
    cudaMemcpy(variableMessages, d_variableMessages, maxFactorDegree*M*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_variableMessages);
    cudaFree(d_factorDegrees);
}

void updateFactorMessagesTanhWrap(int N, int M, double* variableMessages, double* factorMessages, int* factors, 
        int** factorToVariables, int* factorDegrees, int maxFactorDegree, int** factorToPos, int maxVariableDegree)
{
    double *d_variableMessages, *d_factorMessages;
    int *d_factors, *d_factorToVariables, *d_factorDegrees, *d_factorToPos;
    cudaMalloc(&d_variableMessages, maxFactorDegree*M*sizeof(double));
    cudaMalloc(&d_factorMessages, maxVariableDegree*N*sizeof(double));
    cudaMalloc(&d_factors, M*sizeof(int));
    cudaMalloc(&d_factorToVariables, maxFactorDegree*M*sizeof(int));
    cudaMalloc(&d_factorDegrees, M*sizeof(int));
    cudaMalloc(&d_factorToPos, maxFactorDegree*M*sizeof(int));
    cudaMemcpy(d_variableMessages, variableMessages, maxFactorDegree*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorMessages, factorMessages, maxVariableDegree*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factors, factors, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorToVariables, factorToVariables[0], maxFactorDegree*M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorDegrees, factorDegrees, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorToPos, factorToPos[0], maxFactorDegree*M*sizeof(int), cudaMemcpyHostToDevice);
    updateFactorMessagesTanh<<<(M+255)/256,256>>>(M, d_variableMessages, d_factorMessages, d_factors, 
            d_factorToVariables, d_factorDegrees, maxFactorDegree, d_factorToPos, maxVariableDegree);
    cudaMemcpy(factorMessages, d_factorMessages, maxVariableDegree*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_variableMessages);
    cudaFree(d_factorMessages);
    cudaFree(d_factors);
    cudaFree(d_factorToVariables);
    cudaFree(d_factorDegrees);
    cudaFree(d_factorToPos);
}

void updateFactorMessagesMinSum(int alpha, int N, int M, double* variableMessages, double* factorMessages, int* factors,
        int** factorToVariables, int* factorDegrees, int maxFactorDegree, int** factorToPos, int maxVariableDegree)
{
    double *d_variableMessages, *d_factorMessages;
    int *d_factors, *d_factorToVariables, *d_factorDegrees, *d_factorToPos;
    cudaMalloc(&d_variableMessages, maxFactorDegree*M*sizeof(double));
    cudaMalloc(&d_factorMessages, maxVariableDegree*N*sizeof(double));
    cudaMalloc(&d_factors, M*sizeof(int));
    cudaMalloc(&d_factorToVariables, maxFactorDegree*M*sizeof(int));
    cudaMalloc(&d_factorDegrees, M*sizeof(int));
    cudaMalloc(&d_factorToPos, maxFactorDegree*M*sizeof(int));
    cudaMemcpy(d_variableMessages, variableMessages, maxFactorDegree*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorMessages, factorMessages, maxVariableDegree*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factors, factors, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorToVariables, factorToVariables[0], maxFactorDegree*M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorDegrees, factorDegrees, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factorToPos, factorToPos[0], maxFactorDegree*M*sizeof(int), cudaMemcpyHostToDevice);
    updateFactorMessagesMinSum<<<(M+255)/256,256>>>(alpha, M, d_variableMessages, d_factorMessages, d_factors, 
            d_factorToVariables, d_factorDegrees, maxFactorDegree, d_factorToPos, maxVariableDegree);
    cudaMemcpy(factorMessages, d_factorMessages, maxVariableDegree*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_variableMessages);
    cudaFree(d_factorMessages);
    cudaFree(d_factors);
    cudaFree(d_factorToVariables);
    cudaFree(d_factorDegrees);
    cudaFree(d_factorToPos);
}

void updateVariableMessagesWrap(int N, int M, int nQubits, double* factorMessages, double* variableMessages, int** variableToFactors, 
        int* variableDegrees, int maxVariableDegree, int** variableToPos, int maxFactorDegree, int llrp0, int llrq0)
{
    double *d_factorMessages, *d_variableMessages;
    int *d_variableToFactors, *d_variableDegrees, *d_variableToPos;
    cudaMalloc(&d_factorMessages, maxVariableDegree*N*sizeof(double));
    cudaMalloc(&d_variableMessages, maxFactorDegree*M*sizeof(double));
    cudaMalloc(&d_variableToFactors, maxVariableDegree*N*sizeof(int));
    cudaMalloc(&d_variableDegrees, N*sizeof(int));
    cudaMalloc(&d_variableToPos, maxVariableDegree*N*sizeof(int));
    cudaMemcpy(d_factorMessages, factorMessages, maxVariableDegree*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableMessages, variableMessages, maxFactorDegree*M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableToFactors, variableToFactors[0], maxVariableDegree*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableDegrees, variableDegrees, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableToPos, variableToPos[0], maxVariableDegree*N*sizeof(int), cudaMemcpyHostToDevice);
    updateVariableMessages<<<(N+255)/256,256>>>(N, nQubits, d_factorMessages, d_variableMessages, 
            d_variableToFactors, d_variableDegrees, maxVariableDegree, d_variableToPos, maxFactorDegree, llrp0, llrq0);
    cudaMemcpy(variableDegrees, d_variableDegrees, maxFactorDegree*M*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_factorMessages);
    cudaFree(d_variableMessages);
    cudaFree(d_variableToFactors);
    cudaFree(d_variableDegrees);
    cudaFree(d_variableToPos);
}

void calcMarginalsWrap(int N, int nQubits, double* marginals, double* factorMessages, int* variableDegrees, int maxVariableDegree, double llrp0, double llrq0)
{
    double *d_marginals, *d_factorMessages; 
    int *d_variableDegrees;
    cudaMalloc(&d_marginals, N*sizeof(double));
    cudaMalloc(&d_factorMessages, maxVariableDegree*N*sizeof(double));
    cudaMalloc(&d_variableDegrees, N*sizeof(int));
    cudaMemcpy(&d_factorMessages, factorMessages, maxVariableDegree*N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(&d_variableDegrees, variableDegrees, N*sizeof(int), cudaMemcpyHostToDevice);
    calcMarginals<<<(N+255)/256,256>>>(N, nQubits, d_marginals, d_factorMessages, d_variableDegrees, maxVariableDegree, llrp0, llrq0);
    cudaMemcpy(marginals, d_marginals, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_marginals);
    cudaFree(d_factorMessages);
}

void bpCorrectionWrap(int N, int M, int nQubits, int nChecks, double* marginals, 
        int* variables, int* factors, int** variableToFactors, int* variableDegrees, int maxVariableDegree)
{
    double *d_marginals;
    int *d_variables, *d_factors, *d_variableToFactors, *d_variableDegrees;
    cudaMalloc(&d_marginals, N*sizeof(double));
    cudaMalloc(&d_variables, N*sizeof(int));
    cudaMalloc(&d_factors, M*sizeof(int));
    cudaMalloc(&d_variableToFactors, maxVariableDegree*N*sizeof(int));
    cudaMalloc(&d_variableDegrees, N*sizeof(int));
    cudaMemcpy(d_marginals, marginals, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variables, variables, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_factors, factors, M*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableToFactors, variableToFactors[0], maxVariableDegree*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_variableDegrees, variableDegrees, N*sizeof(int), cudaMemcpyHostToDevice);
    bpCorrection<<<(N+255)/256,256>>>(nQubits, nChecks, d_marginals, d_variables, d_factors, 
            d_variableToFactors, d_variableDegrees, maxVariableDegree);
    cudaMemcpy(variables, d_variables, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(factors, d_factors, M*sizeof(int), cudaMemcpyDeviceToHost);
}
