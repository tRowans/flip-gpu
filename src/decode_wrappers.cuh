#ifndef DECODEWRAP_H
#define DECODEWRAP_H

void wipeArrayWrap(int N, int* array);
void applyErrorsWrap(int N, unsigned int seed, int* lookup, int* errorTarget, float errorProb);
void calculateSyndromeWrap(int N, int* lookup, int* qubits, int* syndrome, int** edgeToFaces);
void flipWrap(int N, int* qLookup, int* sLookup, int* qubits, int* syndrome, int** faceToEdges);
void pflipWrap(int N, unsigned int seed, int* qLookup, int* sLookup, int* qubits, int* syndrome, int** faceToEdges);
void updateSyndromeMessagesWrap(int N, int* lookup, int* qubitMessages, int* syndrome, int* syndromeMessages, int** edgeToFaces, int** faceToEdges);
void updateQubitMessagesWrap(int N, int* lookup, int* qubitMessages, int* syndromeMessages, int** faceToEdges, int** edgeToFaces, int p);
void calcMarginalsWrap(int N, int* lookup, int* qubits, int* qubitMessages, int* syndromeMessages, int p);
void bpCorrectionWrap(int N, int* lookup, int* qubits, int* qubitMarginals);
void measureLogicalsWrap(int N, int* lookup, int* qubits, int &nOdd, int L, char bounds);

#endif
