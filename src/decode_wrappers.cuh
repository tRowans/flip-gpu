#ifndef DECODEWRAP_H
#define DECODEWRAP_H

void wipeArrayWrap(int maxIndex, int* array);
void arrayErrorsWrap(int maxIndex, unsigned int seed, int* errorTarget, float errorProb);
void depolErrorsWrap(int maxIndex, unsigned int seed, int* qubitsX, int* qubitsZ, float errorProb);
void calculateSyndromeWrap(int M, int nQubits, int* qubits, int* syndrome, 
        int** factorToVariables, int factorDegrees, int maxFactorDegree);
void flipWrap(int M, int N, int nQubits, int nChecks, int* qubits, int* syndrome, 
        int** variableToFactors, int* variableDegrees, int maxVariableDegree);
void pflipWrap(int M, int N, int nQubits, int nChecks, unsigned int seed, int* qubits, int* syndrome, 
        int** variableToFactors, int* variableDegrees, int maxVariableDegree);
void initVariableMessagesWrap(int M, int nChecks, double** variableMessages, 
        int* factorDegrees, int maxFactorDegree, double llrp0, double llrq0);
void updateFactorMessagesTanhWrap(int M, int N, double** variableMessages, double** factorMessages, int* syndrome, 
        int** factorToVariables, int* factorDegrees, int maxFactorDegree, int** factorToPos, int maxVariableDegree);
void updateFactorMessagesMinSum(int alpha, int M, int N, double** variableMessages, double** factorMessages, int* syndrome
        int** factorToVariables, int* factorDegrees, int maxFactorDegree, int** factorToPos, int maxVariableDegree);
void updateVariableMessagesWrap(int M, int N, int nQubits, double** factorMessages, double** variableMessages, int** variableToFactors, 
        int* variableDegrees, int maxVariableDegree, int** variableToPos, int maxFactorDegree, int llrp0, int llrq0);
void calcMarginalsWrap(int N, int nQubits, double* marginals, double** factorMessages, double llrp0, double llrq0, int maxVariableDegree);
void bpCorrectionWrap(int M, int N, int nQubits, int nChecks, double* marginals, 
        int* qubits, int* syndrome, int** variableToFactors, int* variableDegrees, int maxVariableDegree);

#endif
