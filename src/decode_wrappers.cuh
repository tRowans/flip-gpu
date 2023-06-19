#ifndef DECODEWRAP_H
#define DECODEWRAP_H

void wipeArrayWrap(int maxIndex, int* array);
void depolErrorsWrap(int N_X, int N_Z, int nQubits, unsigned int seed, int* variablesX, int* variablesZ, float errorProb);
void measErrorsWrap(int nQubits, int nChecks, unsigned int seed, int* variables, float errorProb);
void calculateSyndromeWrap(int M, int nQubits, int* variables, int* factors, 
        int** factorToVariables, int* factorDegrees, int maxFactorDegree);
void flipWrap(int N, int M, int nQubits, int* variables, int* factors, 
        int** variableToFactors, int* variableDegrees, int maxVariableDegree);
void subsetFlipWrap(int N, int M, int rangeStart, int rangeEnd, int* variables, int* factors, int** variableToFactors, int* variableDegrees, int maxVariableDegree);
void pflipWrap(int N, int M, int nQubits, unsigned int seed, int* variables, int* factors, 
        int** variableToFactors, int* variableDegrees, int maxVariableDegree);
void initVariableMessagesWrap(int M, int nChecks, double* variableMessages, 
        int* factorDegrees, int maxFactorDegree, double llrp0, double llrq0);
void updateFactorMessagesTanhWrap(int N, int M, double* variableMessages, double* factorMessages, int* factors, 
        int** factorToVariables, int* factorDegrees, int maxFactorDegree, int** factorToPos, int maxVariableDegree);
void updateFactorMessagesMinSum(double alpha, int N, int M, double* variableMessages, double* factorMessages, int* factors,
        int** factorToVariables, int* factorDegrees, int maxFactorDegree, int** factorToPos, int maxVariableDegree);
void updateVariableMessagesWrap(int N, int M, int nQubits, double* factorMessages, double* variableMessages, int** variableToFactors, 
        int* variableDegrees, int maxVariableDegree, int** variableToPos, int maxFactorDegree, double llrp0, double llrq0);
void calcMarginalsWrap(int N, int nQubits, double* marginals, double* factorMessages, 
        int* variableDegrees, int maxVariableDegree, double llrp0, double llrq0);
void bpCorrectionWrap(int N, int M, int nQubits, int nChecks, double* marginals, 
        int* variables, int* factors, int** variableToFactors, int* variableDegrees, int maxVariableDegree);

#endif
