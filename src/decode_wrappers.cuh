#ifndef DECODEWRAP_H
#define DECODEWRAP_H

void wipeArrayWrap(int N, int* array);
void applyErrorsWrap(int N, unsigned int seed, int* lookup, int* errorTarget, double errorProb);
void calculateSyndromeWrap(int N, int* lookup, int* qubits, int* syndrome, int** edgeToFaces);
void flipWrap(int N, int* qLookup, int* sLookup, int* qubits, int* syndrome, int** faceToEdges, int** edgeToFaces, double* variableMessages, double* qubitMarginals);
void pflipWrap(int N, unsigned int seed, int* qLookup, int* sLookup, int* qubits, int* syndrome, int** faceToEdges, int** edgeToFaces, double* variableMessages, double* qubitMarginals);
void initVariableMessagesWrap(int N, int* lookup, double* variableMessages, double llr0, double llrq0);
void updateFactorMessagesWrap(int N, int* lookup, double* variableMessages, int* syndrome, double* factorMessages, int** edgeToFaces, int** faceToEdges);
void updateVariableMessagesWrap(int N, int* lookup, double* variableMessages, double* factorMessages, int** faceToEdges, int** edgeToFaces, double llr0);
void calcMarginalsWrap(int N, int* qLookup, int* sLookup, double* qubitMarginals, double* stabMarginals, double* factorMessages, double llr0, double llrq0);
void calcQubitMarginalsWrap(int N, int* lookup, double* qubitMarginals, double* factorMessages, double llr0);
void calcStabMarginalsWrap(int N, int* lookup, double* stabMarginals, double* factorMessages, double llrq0);
void bpSyndromeCorrectionWrap(int N, int* lookup, int* syndrome, double* factorMessages, double* stabMarginals, int** edgeToFaces, int** faceToEdges);
void bpCorrectionWrap(int N, int* lookup, int* qubits, double* marginals);
void measureLogicalsWrap(int N, int* lookup, int* qubits, int &nOdd, int L, char bounds);

#endif
