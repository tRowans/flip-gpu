#include "code.h"

//-----------------------PRIVATE-----------------------

void Code::readParityCheckMatrix(std::string hFile, std::vector<std::vector<int>> &H)
{
    std::ifstream parityCheck(hFile);
    std::string line;
    while (std::getline(parityCheck, line))
    {
        std::vector<int> row;
        for (int i=0; i<line.size(); i+=2)
        {
            row.push_back(line[i]-'0');
        }
        H.push_back(row);
    }
}

void Code::getVariableDegrees(int M, int N, int* variableDegrees, int& maxVariableDegree, std::vector<std::vector<int>> &H)
{
    maxVariableDegree = 0;
    for (int j=0; j<N; ++j)
    {
        int degree = 0;
        for (int i=0; i<M; ++i) degree += H[i][j];
        variableDegrees[j] = degree;
        if (degree > maxVariableDegree) maxVariableDegree = degree;
    }
}

void Code::getFactorDegrees(int M, int N, int* factorDegrees, int& maxFactorDegree, std::vector<std::vector<int>> &H)
{
    maxFactorDegree = 0;
    for (int i=0; i<M; ++i)
    {
        int degree = 0;
        for (int j=0; j<N; ++j) degree += H[i][j];
        factorDegrees[i] = degree;
        if (degree > maxFactorDegree) maxFactorDegree = degree;
    }
}

void Code::buildVariableToFactors(int M, int N, int** variableToFactors, int maxVariableDegree, std::vector<std::vector<int>> &H)
{
    for (int i=0; i<N*maxVariableDegree; ++i) variableToFactors[0][i] = -1;
    //-1s will get overwritten for all array spaces that represent valid variable -> factor mappings
    //and will persist elsewhere (i.e. if a given variable has degree < maxVariableDegree). 
    //No factor has index -1 so this lets us detect if we accidentally access an invalid array space
    //and it is less annoying than having undefined elements in the array
    for (int j=0; j<N; ++j)
    {
        int factorNumber = 0;
        for (int i=0; i<M; ++i)
        {
            if (H[i][j] == 1)
            {
                variableToFactors[j][factorNumber] = i;
                factorNumber++;
            }
        }
    }
}

void Code::buildFactorToVariables(int M, int N, int** factorToVariables, int maxFactorDegree, std::vector<std::vector<int>> &H)
{
    for (int i=0; i<M*maxFactorDegree; ++i) factorToVariables[0][i] = -1;   //same as above
    for (int i=0; i<M; ++i)
    {
        int variableNumber = 0;
        for (int j=0; j<N; ++j)
        {
            if (H[i][j] == 1)
            {
                factorToVariables[i][variableNumber] = j;
                variableNumber++;
            }
        }
    }
}

void Code::buildNodeToPos(int nNodes, int** nodeToPos, int* nodeDegrees, int** nodeToNeighbours, 
                                int maxNodeDegree, int* neighbourDegrees, int** neighbourToNodes)
{
    for (int i=0; i<nNodes*maxNodeDegree; ++i) nodeToPos[0][i] = -1;
    for (int i=0; i<nNodes; ++i)
    {
        for (int j=0; j<nodeDegrees[i]; ++j)
        {
            int f = nodeToNeighbours[i][j];
            int pos = 0;
            while (pos < neighbourDegrees[f])
            {
                if (neighbourToNodes[f][pos] == i) break;
                else ++pos;
            }
            nodeToPos[i][j] = pos;
        }
    }
}

//-------------------------PUBLIC-------------------------

Code::Code(std::string codename, int n)
{
    readParityCheckMatrix(codename+"_hx.txt", H_X);
    readParityCheckMatrix(codename+"_hz.txt", H_Z);
   
    //Parity check matrix for BP with metachecks has the form
    //
    //         <-----------N------------>
    //            <---n--->  <---m--->
    // 
    //  ^  ^   |- |-     -|  |-     -¾|-|
    //  |  |   |  |       |  |       |  |
    //  |  m   |  |   H   |  |   I   |  |
    //  |  |   |  |       |  |       |  |
    //  |  v   |  |_     _|  |_     _|  |
    //  M      |                        |
    //  |  ^   |  |-     -|  |-     -|  |
    //  |  |   |  |       |  |       |  |
    //  |  x   |  |   0   |  |   C   |  |
    //  |  |   |  |       |  |       |  |
    //  v  v   |_ |_     _|  |_     _| _|
    //
    //where: n = number of qubits
    //       m = number of checks
    //       x = number of metachecks
    //       M = total number of factors
    //       N = total number of variable
    //       H = normal parity check matrix
    //       I = identity matrix of size m
    //       C = metacheck matrix
    //
    //Measurement errors are implemented using the I submatrix
    //so this needs to be added even if we just want to use flip
    //or BP without metachecks. C only needs to be added if we
    //want to use BP with metachecks.

    nQubits = n;
    M_X = H_X.size();
    N_X = H_X[0].size();    
    M_Z = H_Z.size();
    N_Z = H_Z[0].size();   
    //Adding I to XPCM if it doesn't exist
    if (nQubits == N_X)
    {
        N_X = nQubits + M_X;
        nChecksX = M_X;
        for (int i=0; i<nChecksX; ++i)
        {
            for (int j=0; j<nChecksX; ++j)
            {
                H_X[i].push_back(i==j);
            }
        }
    }
    else nChecksX = N_X - nQubits;
    nMetachecksX = M_X - nChecksX;
    //Adding I to ZPCM if it doesn't exist
    if (nQubits == N_Z)
    {
        N_Z = nQubits + M_Z;
        nChecksZ = M_Z;
        for (int i=0; i<nChecksZ; ++i)
        {
            for (int j=0; j<nChecksZ; ++j)
            {
                H_Z[i].push_back(i==j);
            }
        }
    }
    else nChecksZ = N_Z - nQubits;
    nMetachecksZ = M_Z - nChecksZ;
    
    variableDegreesX = new int[N_X];     
    variableDegreesZ = new int[N_Z];     
    factorDegreesX = new int[M_X]; 
    factorDegreesZ = new int[M_Z];

    getVariableDegrees(M_X, N_X, variableDegreesX, maxVariableDegreeX, H_X);
    getVariableDegrees(M_Z, N_Z, variableDegreesZ, maxVariableDegreeZ, H_Z);
    getFactorDegrees(M_X, N_X, factorDegreesX, maxFactorDegreeX, H_X);
    getFactorDegrees(M_Z, N_Z, factorDegreesZ, maxFactorDegreeZ, H_Z);
    
    //This is a trick to get a dynamically allocated 2D array in contiguous memory
    //which means the whole thing can easily be copied to a 1D array on the GPU
    //(GPU does not like 2D arrays)
    variableToFactorsX = new int*[N_X]; 
    variableToFactorsX[0] = new int[N_X*maxVariableDegreeX]; 
    variableToFactorsZ = new int*[N_Z];
    variableToFactorsZ[0] = new int[N_Z*maxVariableDegreeZ];
    factorToVariablesX = new int*[M_X];
    factorToVariablesX[0] = new int[M_X*maxFactorDegreeX];
    factorToVariablesZ = new int*[M_Z];
    factorToVariablesZ[0] = new int[M_Z*maxFactorDegreeZ];
    
    for (int i=1; i<N_X; ++i) variableToFactorsX[i] = variableToFactorsX[i-1] + maxVariableDegreeX; 
    for (int i=1; i<N_Z; ++i) variableToFactorsZ[i] = variableToFactorsZ[i-1] + maxVariableDegreeZ; 
    for (int i=1; i<M_X; ++i) factorToVariablesX[i] = factorToVariablesX[i-1] + maxFactorDegreeX;
    for (int i=1; i<M_Z; ++i) factorToVariablesZ[i] = factorToVariablesZ[i-1] + maxFactorDegreeZ;

    buildVariableToFactors(M_X, N_X, variableToFactorsX, maxVariableDegreeX, H_X);
    buildVariableToFactors(M_Z, N_Z, variableToFactorsZ, maxVariableDegreeZ, H_Z);
    buildFactorToVariables(M_X, N_X, factorToVariablesX, maxFactorDegreeX, H_X);
    buildFactorToVariables(M_Z, N_Z, factorToVariablesZ, maxFactorDegreeZ, H_Z);

    variableToPosX = new int*[N_X];
    variableToPosX[0] = new int[N_X*maxVariableDegreeX];
    variableToPosZ = new int*[N_Z];
    variableToPosZ[0] = new int[N_Z*maxVariableDegreeZ];
    factorToPosX = new int*[M_X];
    factorToPosX[0] = new int[M_X*maxFactorDegreeX];
    factorToPosZ = new int*[M_Z];
    factorToPosZ[0] = new int[M_Z*maxFactorDegreeZ];

    for (int i=1; i<N_X; ++i) variableToPosX[i] = variableToPosX[i-1] + maxVariableDegreeX;
    for (int i=1; i<N_Z; ++i) variableToPosZ[i] = variableToPosZ[i-1] + maxVariableDegreeZ;
    for (int i=1; i<M_X; ++i) factorToPosX[i] = factorToPosX[i-1] + maxFactorDegreeX;
    for (int i=1; i<M_Z; ++i) factorToPosZ[i] = factorToPosZ[i-1] + maxFactorDegreeZ;

    buildNodeToPos(N_X, variableToPosX, variableDegreesX, variableToFactorsX, maxVariableDegreeX, factorDegreesX, factorToVariablesX);
    buildNodeToPos(N_Z, variableToPosZ, variableDegreesZ, variableToFactorsZ, maxVariableDegreeZ, factorDegreesZ, factorToVariablesZ);
    buildNodeToPos(M_X, factorToPosX, factorDegreesX, factorToVariablesX, maxFactorDegreeX, variableDegreesX, variableToFactorsX);
    buildNodeToPos(M_Z, factorToPosZ, factorDegreesZ, factorToVariablesZ, maxFactorDegreeZ, variableDegreesZ, variableToFactorsZ);
}

Code::~Code()
{
    delete[] variableDegreesX;
    delete[] variableDegreesZ;
    delete[] factorDegreesX;
    delete[] factorDegreesZ;

    delete[] variableToFactorsX[0];
    delete[] variableToFactorsX;
    delete[] variableToFactorsZ[0];
    delete[] variableToFactorsZ;
    delete[] factorToVariablesX[0];
    delete[] factorToVariablesX;
    delete[] factorToVariablesZ[0];
    delete[] factorToVariablesZ;

    delete[] variableToPosX[0];
    delete[] variableToPosX;
    delete[] variableToPosZ[0];
    delete[] variableToPosZ;
    delete[] factorToPosX[0];
    delete[] factorToPosX;
    delete[] factorToPosZ[0];
    delete[] factorToPosZ;
}
