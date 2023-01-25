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

void Code::getVariableDegrees(int M, int N, int* variableDegrees, int maxVariableDegree, std::vector<std::vector<int>> &H)
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

void Code::getCheckDegrees(int M, int N, int* factorDegrees, int maxFactorDegree, std::vector<std::vector<int>> &H)
{
    maxFactorDegree = 0;
    for (int i=0; i<M; ++i)
    {
        int degree = 0;
        for (int j=0; j<N; ++j) degree += H[i][j];
        factorDegrees[i] = degree;
        if (degree > maxFactorDegrees) maxFactorDegrees = degree;
    }
}

void Code::buildVariableToFactors(int M, int N, int** variableToFactors, int maxVariableDegree, std::vector<std::vector<int>> &H)
{
    for (int i=0; i<N*maxVariableDegree; ++i) variableToFactors[0][i] = -1;
    //-1s will get overwritten for all array spaces that represent valid variable -> factor mappings
    //and will persist elsewhere (i.e. if a given variable has degree < maxVariableDegree). 
    //No factor has index -1 so this lets us detect if we accidentally access an invalid array space
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

void Code::buildNodeToPos(int nNodes, int** nodeToPos, int* nodeDegrees, int** nodeToNeighbours, int* neighbourDegrees, int** neighbourToNodes)
{
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
   
    //Parity check matrix for BP has the form
    //
    //   .     <-----------N------------>
    //   .        <---n--->  <---m--->
    // 
    //  ^  ^   |‾ |‾     ‾|  |‾     ‾| ‾|
    //  |  |   |  |       |  |       |  |
    //  |. m   |  |   H   |  |   I   |  |
    //  |. |   |  |       |  |       |  |
    //  |. v   |  |_     _|  |_     _|  |
    //  M.     |                        |
    //  |. ^   |  |‾     ‾|  |‾     ‾|  |
    //  |. |   |  |       |  |       |  |
    //  |. x   |  |   0   |  |   C   |  |
    //  |. |   |  |       |  |       |  |
    //  v. v   |_ |_     _|  |_     _| _|
    //
    //where: n = number of qubits
    //       m = number of checks
    //       x = number of metachecks
    //       M = total number of factors
    //       N = total number of variable
    //       H = normal parity check matrix
    //       I = identity matrix of size m
    //       C = metacheck matrix

    nQubits = n;
    H_X.size() = M_X;
    H_X[0].size() = N_X;
    H_Z.size() = M_Z;
    H_Z[0].size() = N_Z
    if (N_X == nQubits) nChecksX = M_X;   //No BP 
    else
    {
        nChecksX = N_X - nQubits;
        nMetachecksX = M_X - nChecksX;
    }
    if (N_Z == nQubits) nChecksZ = M_Z;   //No BP
    else
    {
        nChecksZ = N_Z - nQubits;
        nMetachecksZ = M_Z - nChecksZ;
    }
    
    variableDegreesX = new int[N_X];     
    variableDegreesZ = new int[N_Z];     
    factorDegreesX = new int[M_X]; 
    factorDegreesZ = new int[M_Z];

    getVariableDegrees(M_X, N_X, variableDegreesX, maxVariableDegreeX, H_X)
    getVariableDegrees(M_Z, N_Z, variableDegreesZ, maxVariableDegreeZ, H_Z)
    getFactorDegrees(M_X, N_X, factorDegreesX, maxFactorDegreeX, H_X)
    getFactorDegrees(M_Z, N_Z, factorDegreesZ, maxFactorDegreeZ, H_Z)
    
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

    buildNodeToPos(N_X, variableToPosX, variableDegreesX, variableToFactorsX, factorDegreesX, factorToVariablesX);
    buildNodeToPos(N_Z, variableToPosZ, variableDegreesZ, variableToFactorsZ, factorDegreesZ, factorToVariablesZ);
    buildNodeToPos(M_X, factorToPosX, factorDegreesX, factorToVariablesX, variableDegreesX, variableToFactorsX);
    buildNodeToPos(M_Z, factorToPosZ, factorDegreesZ, factorToVariablesZ, variableDegreesZ, variableToFactorsZ);
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
