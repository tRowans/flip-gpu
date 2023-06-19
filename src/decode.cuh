#ifndef DECODE_H
#define DECODE_H

#include<curand.h>
#include<curand_kernel.h>

__global__ void createStates(int maxIndex, unsigned int seed, curandState_t* states);

__global__ void wipeArray(int maxIndex, int* array);

__global__ void depolErrors(int nQubits, curandState_t* states, int* variablesX, int* variablesZ, float errorProb);

__global__ void measErrors(int nQubits, int nChecks, curandState_t* states, int* variables, float errorProb);

__global__ void calculateSyndrome(int M, int* variables, int* factors, int* factorToVariables, int* factorDegrees, int maxFactorDegree);

__global__ void flip(int nQubits, int* variables, int* factors, int* variableToFactors, int* variableDegrees, int maxVariableDegree);

__global__ void subsetFlip(int rangeStart, int rangeEnd, int* variables, int* factors, int* variableToFactors, int* variableDegrees, int maxVariableDegree);

__global__ void pflip(int nQubits, curandState_t* states, int* variables, int* factors, 
        int* variableToFactors, int* variableDegrees, int maxVariableDegree);

__global__ void initVariableMessages(int M, int nChecks, double* variableMessages, 
        int* factorDegrees, int maxFactorDegree, double llrp0, double llrq0);

__global__ void updateFactorMessagesTanh(int M, double* variableMessages, double* factorMessages, int* factors,
        int* factorToVariables, int* factorDegrees, int maxFactorDegree, int* factorToPos, int maxVariableDegree);

__global__ void updateFactorMessagesMinSum(double alpha, int M, double* variableMessages, double* factorMessages, int* factors,
        int* factorToVariables, int* factorDegrees, int maxFactorDegree, int* factorToPos, int maxVariableDegree);

__global__ void updateVariableMessages(int N, int nQubits, double* factorMessages, double* variableMessages, int* variableToFactors, 
        int* variableDegrees, int maxVariableDegree, int* variableToPos, int maxFactorDegree, double llrp0, double llrq0);

__global__ void calcMarginals(int N, int nQubits, double* marginals, double* factorMessages, 
        int* variableDegrees, int maxVariableDegree, double llrp0, double llrq0);

__global__ void bpCorrection(int nQubits, int nChecks, double* marginals, int* variables, int* factors, 
        int* variableToFactors, int* variableDegrees, int maxVariableDegree);

#endif
