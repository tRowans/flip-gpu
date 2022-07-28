#ifndef DECODEWRAP_H
#define DECODEWRAP_H

void wipeArrayWrap(int N, int* array);
void arrayErrorsWrap(int N, unsigned int seed, int* errorTarget, float errorProb);
void depolErrorsWrap(int N, unsigned int seed, int* qubitsX, int* qubitsZ, float errorProb);
void flipWrap(int N, int M, int* qubits, int* syndrome, int** bitToChecks, int maxBitDegree);
void pflipWrap(int N, int M, unsigned int seed, int* qubits, int* syndrome, int** bitToChecks, int maxBitDegree);
void calculateSyndromeWrap(int N, int M, int* qubits, int* syndrome, int** checkToBits, int maxCheckDegree);

#endif
