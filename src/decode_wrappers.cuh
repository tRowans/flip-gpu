#ifndef DECODEWRAP_H
#define DECODEWRAP_H

void wipeArrayWrap(int N, int* array);
void applyErrorsWrap(int N, unsigned int seed, int* lookup, int* errorTarget, float errorProb);
void flipWrap(int N, int* qLookup, int* sLookup, int* qubits, int* syndrome, int** faceToEdges);
void pflipWrap(int N, unsigned int seed, int* qLookup, int* sLookup, int* qubits, int* syndrome, int** faceToEdges);
void edgeFlipWrap(int N, unsigned int seed, int* qLookup, int* sLookup, 
                      int* qubits, int* syndrome, int** edgeToFaces, int** faceToEdges);
void calculateSyndromeWrap(int N, int* lookup, int* qubits, int* syndrome, int** edgeToFaces);
void measureLogicalsWrap(int N, int* lookup, int* qubits, int &nOdd, int L, char bounds);

#endif
