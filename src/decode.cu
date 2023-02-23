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

__global__
void depolErrors(int nQubits, curandState_t* states, int* variablesX, int* variablesZ, float errorProb)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID < nQubits)
    {
        if (curand_uniform(&states[threadID]) < errorProb)
        {
            double x = curand_uniform(&states[threadID]);
            if (x < 1/3) variablesX[threadID] = variablesX[threadID] ^ 1;
            else if (1/3 <= x && x < 2/3) variablesZ[threadID] = variablesZ[threadID] ^ 1;
            else if (2/3 <= x)
            {
                variablesX[threadID] = variablesX[threadID] ^ 1;
                variablesZ[threadID] = variablesZ[threadID] ^ 1;
            }
        }
    }
}

__global__
void measErrors(int nQubits, int nChecks, curandState_t* states, int* variables, float errorProb)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID < nChecks)
    {
        //This is not a bit flip of the original measurement error value unlike qubit error variables
        //because a qubit error vector {0,1,0,...} should persist between measurement rounds if not affected by errors (or corrections)
        //whereas the measurement error vector is fully redrawn every measurement round.
        if (curand_uniform(&states[threadID]) < errorProb) variables[threadID+nQubits] = 1;
        else variables[threadID+nQubits] = 0;
    }
}

__global__
void calculateSyndrome(int M, int* variables, int* factors, int* factorToVariables, int* factorDegrees, int maxFactorDegree)
{
    //This includes calculating the metacheck syndrome
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per factor
    if (threadID < M) 
    {
        int parity = 0;
        for (int i=0; i<(factorDegrees[threadID]); ++i)   
        {
            int v = factorToVariables[maxFactorDegree*threadID + i]; 
            if (variables[v] == 1) parity = parity ^ 1;
        }
        factors[threadID] = parity;
    }
}

//----------FLIP----------

//Regular deterministic flip
__global__
void flip(int nQubits, int* variables, int* factors, int* variableToFactors, int* variableDegrees, int maxVariableDegree)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (threadID < nQubits)
    {
        int unsatChecks = 0;
        for (int i=0; i<variableDegrees[threadID]; ++i)
        {
            int stab = variableToFactors[maxVariableDegree*threadID+i];
            if (factors[stab] == 1) unsatChecks++;
        }
        if (unsatChecks > variableDegrees[threadID]/2)
        {
            variables[threadID] = variables[threadID] ^ 1;
            for (int i=0; i<variableDegrees[threadID]; ++i)
            {
                int stab = variableToFactors[maxVariableDegree*threadID+i];
                atomicXor(&factors[stab],1);
            }
        }
    }
}

//Probabilistic flip
__global__
void pflip(int nQubits, curandState_t* states, int* variables, int* factors, int* variableToFactors, int* variableDegrees, int maxVariableDegree)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per qubit
    if (threadID < nQubits)
    {
        int unsatChecks = 0;
        for (int i=0; i<variableDegrees[threadID]; ++i)
        {
            int stab = variableToFactors[maxVariableDegree*threadID+i];
            if (factors[stab] == 1) unsatChecks++;
        }
        if (unsatChecks > variableDegrees[threadID]/2)
        {
            variables[threadID] = variables[threadID] ^ 1;
            for (int i=0; i<variableDegrees[threadID]; ++i)
            {
                int stab = variableToFactors[maxVariableDegree*threadID+i];
                atomicXor(&factors[stab],1);
            }
        }
        else if (static_cast<float>(unsatChecks) == static_cast<float>(variableDegrees[threadID])/2)
        {
            if (curand_uniform(&states[threadID]) < 0.5)
            {
                variables[threadID] = variables[threadID] ^ 1;
                for (int i=0; i<variableDegrees[threadID]; ++i)
                {
                    int stab = variableToFactors[maxVariableDegree*threadID + i];
                    atomicXor(&factors[stab],1);
                }
            }
        }
    }
}

//----------BP----------

__global__
void initVariableMessages(int M, int nChecks, double* variableMessages, int* factorDegrees, int maxFactorDegree, double llrp0, double llrq0)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per factor
    if (threadID < nChecks) //factor is a stabiliser
    {
        for (int i=0; i<(factorDegrees[threadID]-1); ++i) 
        {
            variableMessages[maxFactorDegree*threadID+i] = llrp0;    //all except last are qubit error variables
        }
        variableMessages[maxFactorDegree*threadID+factorDegrees[threadID]-1] = llrq0;  //last is a measurement error variable
    }
    else if (threadID < M) //factor is a metacheck
    {
        for (int i=0; i<factorDegrees[threadID]; ++i)
        {
            variableMessages[maxFactorDegree*threadID+i] = llrq0;   //all connected variables are for measurement errors
        }
    }
}

//Keeping the two update rules as separate functions for now but can combine them later if it would be easier
__global__
void updateFactorMessagesTanh(int M, double* variableMessages, double* factorMessages, int* factors, 
        int* factorToVariables, int* factorDegrees, int maxFactorDegree, int* factorToPos, int maxVariableDegree)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per factor 
    int degree = factorDegrees[threadID];
    if (threadID < M)
    {
        for (int i=0; i<degree; ++i)
        {
            double m = 1.0;
            for (int j=0; j<degree; ++j)
            {
                if (i != j) m = m*tanh(variableMessages[maxFactorDegree*threadID+j]/2);
            }
            m = (1-2*factors[threadID])*2*atanh(m);

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
void updateFactorMessagesMinSum(double alpha, int M, double* variableMessages, double* factorMessages, int* factors,
        int* factorToVariables, int* factorDegrees, int maxFactorDegree, int* factorToPos, int maxVariableDegree)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per factor
    int degree = factorDegrees[threadID];
    if (threadID < M)
    {
        for (int i=0; i<degree; ++i)
        {
            double m = 1.0;
            double min_message = 1e6;  //just needs to be larger than any reasonable real message
            for (int j=0; j<degree; ++j)
            {
                if (i != j)
                {
                    double message = variableMessages[maxFactorDegree*threadID+j];
                    if (message < min_message) min_message = message;
                    m = m*((message>0) - (message<0));
                }
            }
            m = (1-2*factors[threadID])*alpha*min_message*m;
            int v = factorToVariables[maxFactorDegree*threadID+i]; //message recipient
            int pos = factorToPos[maxFactorDegree*threadID+i];     //position in recipients variableToFactor map
            factorMessages[maxVariableDegree*v+pos] = m;
        }
    }
}

__global__
void updateVariableMessages(int N, int nQubits, double* factorMessages, double* variableMessages, int* variableToFactors,
        int* variableDegrees, int maxVariableDegree, int* variableToPos, int maxFactorDegree, double llrp0, double llrq0)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per variable
    int degree = variableDegrees[threadID];
    if (threadID < N)     
    {
        for (int i=0; i<degree; ++i)
        {
            double m;
            if (threadID < nQubits) m = llrp0;   //messages from qubit error variables
            else m = llrq0;                      //messages from measurement error variables
            for (int j=0; j<degree; ++j) 
            {
                if (i != j) m = m + factorMessages[maxVariableDegree*threadID+j];
            }
            int f = variableToFactors[maxVariableDegree*threadID+i];    //message recipient
            int pos = variableToPos[maxVariableDegree*threadID+i];      //position in recipients factorToVariables map
            variableMessages[maxFactorDegree*f+pos] = m;
        }
    }
}

__global__
void calcMarginals(int N, int nQubits, double* marginals, double* factorMessages, int* variableDegrees, int maxVariableDegree, double llrp0, double llrq0)
{
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; //One thread per variable
    int degree = variableDegrees[threadID];
    if (threadID < N)
    {
        for (int i=0; i<degree; ++i)
        {
            double m;
            if (threadID < nQubits) m = llrp0;
            else m = llrq0;
            for (int j=0; j<degree; ++j) m = m + factorMessages[maxVariableDegree*threadID+j];
            marginals[threadID] = m;
        }
    }
}

__global__
void bpCorrection(int nQubits, int nChecks, double* marginals, int* variables, int* factors, 
        int* variableToFactors, int* variableDegrees, int maxVariableDegree)
{
    //One thread per qubit/stabiliser (do one first and then the other)
    int threadID = blockIdx.x * blockDim.x + threadIdx.x; 
    if (threadID < nChecks)
    {
        //marginals is all qubit error nodes then all measurement error nodes
        if (marginals[nQubits+threadID] < 0) atomicXor(&factors[threadID],1);
    }
    if (threadID < nQubits)
    {
        if (marginals[threadID] < 0)
        {
            variables[threadID] = variables[threadID] ^ 1;
            //update syndrome based on qubit flips
            for (int i=0; i<variableDegrees[threadID]; ++i)
            {
                int f = variableToFactors[maxVariableDegree*threadID+i];
                atomicXor(&factors[f],1);
            }
        }
    }
}
