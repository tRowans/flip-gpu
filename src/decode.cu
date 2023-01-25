#include "code.h"
#include "decode.cuh"

//----------GENERAL----------

__global__
void createStates(int maxIndex, unsigned int seed, curandState_t* states)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per state
    if (threadID < maxIndex)
    {
        curand_init(seed, threadID, 0, &states[threadID]);
    }
}

__global__
void wipeArray(int maxIndex, int* array) 
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per array element
    if (threadID < maxIndex) array[threadID] = 0;
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
void depolErrors(int nQubits, curandState_t* states, int* qubitsX, int* qubitsZ, float errorProb)
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

__global__
void calculateSyndrome(int nChecks, int* qubits, int* syndrome, int* factorToVariables, int* factorDegrees, int maxFactorDegree)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per stabiliser
    if (threadID < nChecks) //first nChecks elements of factorToVariables are stabilisers (others are metachecks)
    {
        int parity = 0;
        for (int i=0; i<(factorDegrees[threadID]-1); ++i)   //-1 because the last one is a measurement error variable node
        {
            int bit = factorToVariables[maxFactorDegree*threadID + i]; 
            if (qubits[bit] == 1) parity = parity ^ 1;
            syndrome[threadID] = parity;
        }
    }
}

//----------FLIP----------

//Regular deterministic flip
__global__
void flip(int nQubits, int* qubits, int* syndrome, int* variableToFactors, int* variableDegrees, int maxVariableDegree)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (threadID < nQubits)
    {
        int unsatChecks = 0;
        for (int i=0; i<variableDegrees[threadID]; ++i)
        {
            int stab = variableToFactors[maxVariableDegree*threadID+i];
            if (syndrome[stab] == 1) unsatChecks++;
        }
        if (unsatChecks > variableDegrees[threadID]/2)
        {
            qubits[threadID] = qubits[threadID] ^ 1;
            for (int i=0; i<variableDegrees[threadID]; ++i)
            {
                int stab = variableToFactors[maxVariableDegree*threadID+i];
                atomicXor(&syndrome[stab],1);
            }
        }
    }
}

//Probabilistic flip
__global__
void pflip(int nQubits, curandState_t* states, int* qubits, int* syndrome, int* variableToFactors, int* variableDegrees, int maxVariableDegree)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (threadID < nQubits)
    {
        int unsatChecks = 0;
        for (int i=0; i<variableDegrees[threadID]; ++i)
        {
            int stab = variableToFactors[maxVariableDegree*threadID+i];
            if (syndrome[stab] == 1) unsatChecks++;
        }
        if (unsatChecks > variableDegrees[threadID]/2)
        {
            qubits[threadID] = qubits[threadID] ^ 1;
            for (int i=0; i<variableDegrees[threadID]; ++i)
            {
                int stab = variableToFactors[maxVariableDegree*threadID+i];
                atomicXor(&syndrome[stab],1);
            }
        }
        else if (static_cast<float>(unsatChecks) == static_cast<float>(variableDegrees[threadID])/2)
        {
            if (curand_uniform(&states[threadID]) < 0.5)
            {
                qubits[threadID] = qubits[threadID] ^ 1;
                for (int i=0; i<variableDegrees[threadID]; ++i)
                {
                    int stab = variableToFactors[maxVariableDegree*threadID + i];
                    atomicXor(&syndrome[stab],1);
                }
            }
        }
    }
}

//----------BP----------

__global__
void initVariableMessages(int M, int nChecks, double* variableMessages, int* factorDegrees, int maxFactorDegree, double llr0, double llrq0)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per factor
    if (threadID < nChecks) //factor is a stabiliser
    {
        for (int i=0; i<(factorDegrees[threadID]-1); ++i) 
        {
            variableMessages[maxFactorDegree*threadID+i] = llr0;    //all except last are qubit error variables
        }
        variableMessages[maxFactorDegree*threadID+factorDegrees[threadID]] = llrq0;  //last is a measurement error variable
    }
    else if (threadID < M) //factor is a metacheck
    {
        for (int i=0; i<factorDegrees[threadID]; ++i)
        {
            variableMessages[maxFactorDegree*threadID+i] = llrq0;   //all connected variables are for measurement errors
        }
    }
}

//Conditionals slow down GPU so faster to have separate functions for separate update rules 
__global__
void updateFactorMessagesTanh(int M, double* variableMessages, double* factorMessages, int* syndrome, 
        int* factorToVariables, int* factorDegrees, int maxFactorDegree, int* factorToPos, int maxVariableDegree)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per factor 
    int degree = factorDegrees[threadID]
    if (threadID < M)
    {
        for (int i=0; i<degree; ++i)
        {
            double m = 1.0;
            //if statements are bad for GPU so this is better than one loop that checks (i!=j)
            for (int j=0; j<i; ++j) m = m*tanh(variableMessages[maxFactorDegree*threadID+j]/2);
            for (int j=i+1; j<degree; ++j) m = m*tanh(variableMessages[maxFactorDegree*threadID+j]/2);
            m = (1-2*syndrome[threadID])*2*atanh(m);

            //FactorMessages is organised by which variable the messages are going to, not which factor they come from. 
            //Each variable recieves one message from each adjacent factor and these are in the same order as the order
            //of factors in variableToFactors, so when a check sends a message it needs to know its own place in this order
            //so it can write to the right place

            int v = factorToVariables[maxFactorDegree*threadID+i]; //message recipient
            int pos = factorToPos[maxFactorDegree*threadID+i];     //position in recipients variableToFactor map
            factorMessages[maxVariableDegree*v+pos] = m;
        }
    }
}

__global__
void updateFactorMessagesMinSum(int alpha, int M, double* variableMessages, double* factorMessages, int* syndrome,
        int* factorToVariables, int* factorDegrees, int maxFactorDegree, int* factorToPos, int maxVariableDegree)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per factor
    int degree = factorDegrees[threadID];
    if (threadID < M)
    {
        for (int i=0; i<degree; ++i)
        {
            //MIN SUM RULE
        }

        int v = factorToVariables[maxFactorDegree*threadID+i]; //message recipient
        int pos = factorToPos[maxFactorDegree*threadID+i];     //position in recipients variableToFactor map
        factorMessages[maxVariableDegree*v+pos] = m;
    }
}

}

__global__
void updateVariableMessages(nvar)
