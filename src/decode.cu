#include "code.h"
#include "decode.cuh"

//----------GENERAL----------

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
void applyErrors(int* lookup, curandState_t* states, int* errorTarget, double errorProb)
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

//----------FLIP----------

//Regular deterministic flip
__global__
void flip(int* qLookup, int* sLookup, int* qubits, int* syndrome, int* faceToEdges)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (qLookup[threadID] == 1 && qubitMarginals[threadID] < 0)
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
    if (qLookup[threadID] == 1 && qubitMarginals[threadID] < 0)
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

//----------BP----------

__global__
void initVariableMessages(int* lookup, double* variableMessages, double llr0, double llrq0)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per factor
    if (lookup[threadID] == 1)
    {
        for (int i=0; i<4; ++i) variableMessages[5*threadID+i] = llr0;
        variableMessages[5*threadID+4] = llrq0;
    }
}

__global__
void updateFactorMessages(int* lookup, double* variableMessages, int* syndrome, double* factorMessages, int* edgeToFaces, int* faceToEdges, int N)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per factor (stabiliser)
    if (lookup[threadID] == 1)
    {
        for (int i=0; i<5; ++i)
        {
            double m = 1.0;
            for (int j=0; j<5; ++j)
            {
                 if (i!=j) m = m*tanh(variableMessages[5*threadID+j]/2);
            }
            if (syndrome[threadID] == 0) m = 2*atanh(m);
            else m = -2*atanh(m);

            //FactorMessages is organised by which variable the messages are going to, not which factor they come from.
            if (i < 4)
            {
                //Each qubit recieves 4 messages (one per stabiliser) and these are in the same order as the order of stabilisers in faceToEdges
                //so when a stabiliser sends a message it needs to know its own place in this order so it can write to the right place
                int q = edgeToFaces[4*threadID+i];  //message recipient 
                int pos = 0;
                while (pos < 4)
                {
                    if (faceToEdges[4*q+pos] == threadID) break;    //find relative position of stabiliser in qubit's neighbour lookup
                    else ++pos;
                }
                //write message to appropriate position in factorMessages
                factorMessages[4*q+pos] = m;
            }
            else
            {
                //measurement error variables have only one neighbour so recieve only one message
                factorMessages[4*N+threadID] = m;
            }
        }
    }
}

__global__
void updateVariableMessages(int* lookup, double* variableMessages, double* factorMessages, int* faceToEdges, int* edgeToFaces, double llr0)
{
    //One thread per qubit
    //don't need threads for measurement error varaibles nodes
    //because messages from these don't change as they only have one neighbour
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; 
    if (lookup[threadID] == 1)
    {
        //qubit messages
        for (int i=0; i<4; ++i)
        {
            double m = llr0;
            for (int j=0; j<4; ++j)
            {
                if (i!=j) m = m + factorMessages[4*threadID+j];
            }
            //variableMessages is organised by which factor the messages are going to 
            //(in the same way that factorMessages is organised by variables) 
            //so we have to do the same process as above to find the right place to write the messages
            int s = faceToEdges[4*threadID+i];
            int pos = 0;
            while (pos < 4)
            {
                if (edgeToFaces[4*s+pos] == threadID) break;
                else ++pos;
            }
            variableMessages[5*s+pos] = m;    //every 5th position is for measurement error variable messages so skip these
        }
    }
}

//Joint function to calculate all marginals at once. More efficient than doing separately when running pure BP
__global__
void calcMarginals(int* qLookup, int* sLookup, double* qubitMarginals, double* stabMarginals, double* factorMessages, double llr0, double llrq0, int N)
{
    //First calculates a qubit error marginal for qubit i
    //and then calculates a measurement error marginal for stabiliser j
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; 
    //qubit error marginals
    if (qLookup[threadID] == 1)
    {
        double m = llr0;
        for (int i=0; i<4; ++i) m = m + factorMessages[4*threadID+i];
        qubitMarginals[threadID] = m;
    }
    //measurement error marginals
    if (sLookup[threadID] == 1)
    {
        stabMarginals[threadID] = llrq0 + factorMessages[4*N+threadID];
    }
}

__global__
void bpCorrection(int* qLookup, int* sLookup, int* qubits, double* qubitMarginals, int* syndrome, double* stabMarginals, int* faceToEdges)
{
    //one thread per qubit/stabiliser (do one first then the other)
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (sLookup[threadID] == 1)
    {
        if (stabMarginals[threadID] < 0) syndrome[threadID] = syndrome[threadID] ^ 1;
    }
    if (qLookup[threadID] == 1)
    {
        if (qubitMarginals[threadID] < 0) qubits[threadID] = qubits[threadID] ^ 1;
        //update syndrome based on qubit flips
        for (int i=0; i<4; ++i)
        {
            int e = faceToEdges[4*threadID+i];
            syndrome[e] = syndrome[e] ^ 1;
        }
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
