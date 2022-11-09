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
void flip(int* qLookup, int* sLookup, int* qubits, int* syndrome, int* faceToEdges, int* edgeToFaces, float* qubitMessages, float* qubitMarginals)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (qLookup[threadID] == 1 && qubitMarginals[threadID] > 0.5)
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
                //invert previously calculated qubit messages
                int pos = 0;
                while (pos < 4)
                {
                    if (edgeToFaces[4*stab+pos] == threadID) break;
                    else ++pos;
                }
                qubitMessages[8*stab+2*pos] = 1 - qubitMessages[8*stab+2*pos];
                qubitMessages[8*stab+2*pos+1] = 1 - qubitMessages[8*stab+2*pos+1];
            }
        }
    }
}

//Probabilistic flip
__global__
void pflip(int* qLookup, int* sLookup, int* qubits, int* syndrome, int* faceToEdges, 
            int* edgeToFaces, float* qubitMessages, float* qubitMarginals, curandState_t* states)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (qLookup[threadID] == 1 && qubitMarginals[threadID] > 0.5)
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
                //invert previously calculated qubit messages
                int pos = 0;
                while (pos < 4)
                {
                    if (edgeToFaces[4*stab+pos] == threadID) break;
                    else ++pos;
                }
                qubitMessages[8*stab+2*pos] = 1 - qubitMessages[8*stab+2*pos];
                qubitMessages[8*stab+2*pos+1] = 1 - qubitMessages[8*stab+2*pos+1];
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
                    int pos = 0;
                    while (pos < 4)
                    {
                        if (edgeToFaces[4*stab+pos] == threadID) break;
                        else ++pos;
                    }
                    qubitMessages[8*stab+2*pos] = 1 - qubitMessages[8*stab+2*pos];
                    qubitMessages[8*stab+2*pos+1] = 1 - qubitMessages[8*stab+2*pos+1];
                }
            }
        }
    }
}

//----------BP----------

__global__
void updateSyndromeMessages(int* lookup, float* qubitMessages, int* syndrome, float* syndromeMessages, int* edgeToFaces, int* faceToEdges)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per stabiliser
    if (lookup[threadID] == 1)
    {
        float nMessages[8];    //messages from neighbouring qubits
        for (int i=0; i<8; ++i) nMessages[i] = qubitMessages[8*threadID+i];  
        int seq[12] = {1,2,3,0,2,3,0,1,3,0,1,2};   //GPU doesn't like 2D stuff but this should really be 4x3
        for (int i=0; i<4; ++i)
        {
            //non-target qubits = {{0,0,0},{1,1,0},{1,0,1},{0,1,1}}, this is the message if syndrome == target qubit 
            double m0 = nMessages[2*seq[3*i]+0]*nMessages[2*seq[3*i+1]+0]*nMessages[2*seq[3*i+2]+0]
                       +nMessages[2*seq[3*i]+1]*nMessages[2*seq[3*i+1]+1]*nMessages[2*seq[3*i+2]+0]
                       +nMessages[2*seq[3*i]+1]*nMessages[2*seq[3*i+1]+0]*nMessages[2*seq[3*i+2]+1]
                       +nMessages[2*seq[3*i]+0]*nMessages[2*seq[3*i+1]+1]*nMessages[2*seq[3*i+2]+1];

            //non-target qubits = {{1,0,0},{0,1,0},{0,0,1},{1,1,1}}, this is the message if syndrome != target qubit
            double m1 = nMessages[2*seq[3*i]+1]*nMessages[2*seq[3*i+1]+0]*nMessages[2*seq[3*i+2]+0]
                       +nMessages[2*seq[3*i]+0]*nMessages[2*seq[3*i+1]+1]*nMessages[2*seq[3*i+2]+0]
                       +nMessages[2*seq[3*i]+0]*nMessages[2*seq[3*i+1]+0]*nMessages[2*seq[3*i+2]+1]
                       +nMessages[2*seq[3*i]+1]*nMessages[2*seq[3*i+1]+1]*nMessages[2*seq[3*i+2]+1];

            //syndromeMessages is organised by which qubit the messages are going to, not which stabiliser they come from.
            //Each qubit recieves 8 messages (two per stabiliser, corresponding to inferred values of 0 and 1 respectively). 
            //These four pairs are organised in the same order as the order of stabilisers in faceToEdges
            //so when a stabiliser sends a message it needs to know its own place in this order so it can write to the right place
            int q = edgeToFaces[4*threadID+i];  //message recipient 
            int pos = 0;
            while (pos < 4)
            {
                if (faceToEdges[4*q+pos] == threadID) break;    //find relative position of stabiliser in qubit's neighbour lookup
                else ++pos;
            }
            //write message to appropriate position in syndromeMessages
            if (syndrome[threadID] == 0)
            {
                syndromeMessages[8*q+(2*pos)] = m0;
                syndromeMessages[8*q+(2*pos+1)] = m1;
            }
            else if (syndrome[threadID] == 1)
            {
                syndromeMessages[8*q+(2*pos)] = m1;
                syndromeMessages[8*q+(2*pos+1)] = m0;
            }
        }
    }
}

__global__
void updateQubitMessages(int* lookup, float* qubitMessages, float* syndromeMessages, int* faceToEdges, int* edgeToFaces, float p)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (lookup[threadID] == 1)
    {
        float nMessages[8];  //messages from neighbouring stabilisers
        for (int i=0; i<8; ++i) nMessages[i] = syndromeMessages[8*threadID+i];
        int seq[12] = {1,2,3,0,2,3,0,1,3,0,1,2};    //as above
        for (int i=0; i<4; ++i)
        {
            float m0 = (1-p)*nMessages[2*seq[3*i]]*nMessages[2*seq[3*i+1]]*nMessages[2*seq[3*i+2]];
            float m1 = p*nMessages[2*seq[3*i]+1]*nMessages[2*seq[3*i+1]+1]*nMessages[2*seq[3*i+2]+1];
            //renormalise
            float tot = m0 + m1;
            m0 = m0/tot;
            m1 = m1/tot;
            //qubitMessages is organised by which stabiliser the messages are going to 
            //(in the same way that syndrome messages is organised by qubits) 
            //so we have to do the same process as above to find the right place to write the messages
            //Even though these probabilities are normalised to sum to 1 (and so it is redundant to store them both)
            //we store them both anyway because it is helpful to be able to retrieve either one directly 
            //without conditional calculations when calculating the syndrome messages
            int s = faceToEdges[4*threadID+i];
            int pos = 0;
            while (pos < 4)
            {
                if (edgeToFaces[4*s+pos] == threadID) break;
                else ++pos;
            }
            qubitMessages[8*s+(2*pos)] = m0;
            qubitMessages[8*s+(2*pos+1)] = m1;
        }
    }
}

__global__
void calcMarginals(int* lookup, float* qubitMarginals, float* syndromeMessages, float p)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (lookup[threadID] == 1)
    {
        float m0 = (1-p);
        float m1 = p;
        for (int i=0; i<4; ++i)
        {
            m0 = m0*syndromeMessages[8*threadID+2*i];
            m1 = m1*syndromeMessages[8*threadID+2*i+1];
        }
        //renormalise
        float tot = m0 + m1;
        //only store the probability of an error here
        m1 = m1/tot;
        qubitMarginals[threadID] = m1;
    }
}

__global__
void bpCorrection(int* lookup, int* qubits, float* qubitMarginals)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //one thread per qubit
    if (lookup[threadID] == 1)
    {
        //Don't need to update the syndrome here because this is the final step of the calculation
        if (qubitMarginals[threadID] > 0.5) qubits[threadID] = qubits[threadID] ^ 1;
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
