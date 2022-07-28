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
void arrayErrors(int maxIndex, curandState_t* states, int* errorTarget, float errorProb)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per errorTarget element
    if (threadID < maxIndex)
    {
        if (curand_uniform(&states[threadID]) < errorProb) 
        {
            errorTarget[threadID] = errorTarget[threadID] ^ 1;
        }
    }
}

__global__
void depolErrors(int N, curandState_t* states, int* qubitsX, int* qubitsZ, float errorProb)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID < N)
    {
        if (curand_uniform(&states[threadID]) < errorProb)
        {
            double x = curand_uniform(&states[threadID]);
            if (x < 1/3) qubitsX[threadID] = qubitsX[threadID] ^ 1;
            else if (1/3 <= x && x < 2/3) qubitsZ[threadID] = qubitsZ[threadID] ^ 1;
            else if (2/3 <= x)
            {
                qubitsX[threadID] = qubitsX[threadID] ^ 1;
                qubitsZ[threadID] = qubitsZ[threadID] ^ 1;
            }
        }
    }
}

//Regular deterministic flip
__global__
void flip(int N, int M, int* qubits, int* syndrome, int* bitToChecks, int maxBitDegree)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (threadID < N)
    {
        int totalChecks = 0;
        int unsatChecks = 0;
        for (int i=0; i<maxBitDegree; ++i)
        {
            int stab = bitToChecks[maxBitDegree*threadID+i];
            if (stab != -1)
            {
                totalChecks++;
                if (syndrome[stab] == 1) unsatChecks++;
            }
        }
        if (unsatChecks > totalChecks/2)
        {
            qubits[threadID] = qubits[threadID] ^ 1;
            for (int i=0; i<maxBitDegree; ++i)
            {
                int stab = bitToChecks[maxBitDegree*threadID+i];
                if (stab != -1 && stab < M)
                {
                    atomicXor(&syndrome[stab],1);
                }
            }
        }
    }
}

//Probabilistic flip
__global__
void pflip(int N, int M, curandState_t* states, int* qubits, int* syndrome, int* bitToChecks, int maxBitDegree)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (threadID < N)
    {
        int totalChecks = 0;
        int unsatChecks = 0;
        for (int i=0; i<maxBitDegree; ++i)
        {
            int stab = bitToChecks[maxBitDegree*threadID+i];
            if (stab != -1)
            {
                totalChecks++;
                if (syndrome[stab] == 1) unsatChecks++;
            }
        }
        if (unsatChecks > totalChecks/2)
        {
            qubits[threadID] = qubits[threadID] ^ 1;
            for (int i=0; i<maxBitDegree; ++i)
            {
                int stab = bitToChecks[maxBitDegree*threadID+i];
                if (stab != -1 && stab < M)
                {
                    atomicXor(&syndrome[stab],1);
                }
            }
        }
        else if (static_cast<float>(unsatChecks) == static_cast<float>(totalChecks)/2)
        {
            if (curand_uniform(&states[threadID]) < 0.5)
            {
                qubits[threadID] = qubits[threadID] ^ 1;
                for (int i=0; i<maxBitDegree; ++i)
                {
                    int stab = bitToChecks[maxBitDegree*threadID + i];
                    if (stab != -1 && stab < M)
                    {
                        atomicXor(&syndrome[stab],1);
                    }
                }
            }
        }
    }
}

__global__
void calculateSyndrome(int M,  int* qubits, int* syndrome, int* checkToBits, int maxCheckDegree)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per stabiliser
    if (threadID < M)
    {
        int parity = 0;
        for (int i=0; i<maxCheckDegree; ++i)
        {
            int bit = checkToBits[maxCheckDegree*threadID + i]; 
            if (bit != -1)
            {
                if (qubits[bit] == 1) parity = parity ^ 1;
            }
            syndrome[threadID] = parity;
        }
    }
}
