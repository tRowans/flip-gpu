#include "code.h"
#include "decode.cuh"

__global__
void createStates(int N, unsigned int seed, curandState_t* states)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per state
    //Don't need to use the lookups here b.c. it doesn't matter if we create too many states
    if (threadID < N)
    {
        curand_init(seed, threadID, 0, &states[threadID]);
    }
}

__global__
void wipeArray(int N, int* array) 
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per array element
    //Don't need lookups here either
    if (threadID < N) array[threadID] = 0;
}

//This works for qubit or syndrome errors
//errorTarget is either qubits or syndrome
//errorProb is p or q
__global__
void applyErrors(int* lookup, curandState_t* states, int* errorTarget, float errorProb)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per errorTarget element
    if (lookup[threadID] == 1)
    {
        if (curand_uniform(&states[threadID]) < errorProb) 
        {
            errorTarget[threadID] = errorTarget[threadID] ^ 1;
        }
    }
}

//Regular deterministic flip
__global__
void flip(int* qLookup, int* sLookup, int* qubits, int* syndrome, int* faceToEdges)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (qLookup[threadID] == 1)
    {
        int n = 0;
        for (int i=0; i<4; ++i)
        {
            //faceToEdges is a flat array on the gpu
            if (syndrome[faceToEdges[4*threadID+i]] == 1) n++;
        }
        if (n > 2)
        {
            qubits[threadID] = qubits[threadID] ^ 1;
            for (int i=0; i<4; ++i)
            {
                int stab = faceToEdges[4*threadID+i];
                if (sLookup[stab] == 1) atomicXor(&syndrome[stab],1);
            }
        }
    }
}

//Probabilistic flip
__global__
void pflip(int* qLookup, int* sLookup, int* qubits, int* syndrome, int* faceToEdges, curandState_t* states)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (qLookup[threadID] == 1)
    {
        int n = 0;
        for (int i=0; i<4; i++)
        {
            if (syndrome[faceToEdges[4*threadID+i]] == 1) n++;
        }
        if (n > 2) 
        {
            qubits[threadID] = qubits[threadID] ^ 1;
            for (int i=0; i<4; ++i)
            {
                int stab = faceToEdges[4*threadID+i];
                if (sLookup[stab] == 1) atomicXor(&syndrome[stab],1);
            }
        }
        else if (n == 2) 
        {
            if (curand_uniform(&states[threadID]) < 0.5)
            {
                qubits[threadID] = qubits[threadID] ^ 1;
                for (int i=0; i<4; ++i)
                {
                    int stab = faceToEdges[4*threadID+i];
                    if (sLookup[stab] == 1) atomicXor(&syndrome[stab],1);
                }
            }
        }
    }
}

//Mess up edges
__global__
void edgeFlip(int* qLookup, int* sLookup, int* qubits, int* syndrome, int* edgeToFaces, int* faceToEdges, curandState_t* states)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per stabiliser
    if (sLookup[threadID] == 1 && syndrome[threadID] == 1 && curand_uniform(&states[threadID]) < 0.25)
    {
        for (int i=0; i<4; ++i)
        {
            int qubit = edgeToFaces[4*threadID+i];
            if (qLookup[qubit] == 1)
            {
                atomicXor(&qubits[qubit],1);
                for (int j=0; j<4; ++j)
                {
                    int stab = faceToEdges[4*qubit+j];
                    if (sLookup[stab] == 1) atomicXor(&syndrome[stab],1);
                }
            }
        }
    }
}

__global__
void calculateSyndrome(int* lookup, int* qubits, int* syndrome, int* edgeToFaces)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per stabiliser
    if (lookup[threadID] == 1)
    {
        int parity = 0;
        for (int i=0; i<4; i++)
        {
            if (qubits[edgeToFaces[4*threadID+i]] == 1) parity = parity ^ 1;
        }
        syndrome[threadID] = parity;
    }
}

__global__
void measureLogicals(int* lookup, int* qubits, int* nOdd, int L, char bounds)
{
    //Just check the reps that run in the Z direction
    //Code and error model are symmetric along all axis
    //so expect performance for other two logical qubits is the same
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per logical op rep
    if (lookup[threadID] == 1)
    {
        int qubit = threadID;
        int parity = qubits[qubit];
        //Don't need to check bounds is 'o' or 'c' only here because it was checked earlier
        if (bounds == 'o')
        {
            for (int i=0; i<L-3; ++i)
            {
                qubit += 3*L*L;
                parity = (parity + qubits[qubit]) % 2;
            }
        }
        else
        {
            for (int i=0; i<L; ++i)
            { 
                qubit += 3*L*L;
                parity = (parity + qubits[qubit]) % 2;
            }
        }
        atomicAdd(nOdd, parity);
    }
}
