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

void Code::findMaxBitDegree(std::vector<std::vector<int>> &H, int &maxBitDegree)
{
    maxBitDegree = 0;
    for (int j=0; j<H[0].size(); ++j)
    {
        int weight = 0;
        for (int i=0; i<H.size(); ++i) weight += H[i][j];
        if (weight > maxBitDegree) maxBitDegree = weight;
    }
}

void Code::findMaxCheckDegree(std::vector<std::vector<int>> &mat, int &maxCheckDegree)
{
    maxCheckDegree = 0;
    for (int i=0; i<mat.size(); ++i)
    {
        int weight = 0;
        for (int j=0; j<mat[0].size(); ++j) weight += mat[i][j];
        if (weight > maxCheckDegree) maxCheckDegree = weight;
    }
}

void Code::buildBitToChecks(int M, int** bitToChecks, int maxBitDegree, std::vector<std::vector<int>> &H)
{
    for (int i=0; i<N*maxBitDegree; ++i) bitToChecks[0][i] = -1;
    for (int j=0; j<N; ++j)
    {
        int checkNumber = 0;
        for (int i=0; i<M; ++i)
        {
            if (H[i][j] == 1)
            {
                bitToChecks[j][checkNumber] = i;
                checkNumber++;
            }
        }
    }
}

void Code::buildCheckToBits(int M, int** checkToBits, int maxCheckDegree, std::vector<std::vector<int>> &H)
{
    for (int i=0; i<M*maxCheckDegree; ++i) checkToBits[0][i] = -1;
    for (int i=0; i<M; ++i)
    {
        int bitNumber = 0;
        for (int j=0; j<N; ++j)
        {
            if (H[i][j] == 1)
            {
                checkToBits[i][bitNumber] = j;
                bitNumber++;
            }
        }
    }
}

//-------------------------PUBLIC-------------------------

Code::Code(std::string codename)
{
    readParityCheckMatrix(codename+"_hx.txt", H_X);
    readParityCheckMatrix(codename+"_hz.txt", H_Z);
    findMaxBitDegree(H_X, maxBitDegreeX);
    findMaxBitDegree(H_Z, maxBitDegreeZ);
    findMaxCheckDegree(H_X, maxCheckDegreeX);
    findMaxCheckDegree(H_Z, maxCheckDegreeZ);
    N = H_X[0].size();
    M_X = H_X.size();
    M_Z = H_Z.size();

    //This is a trick to get a dynamically allocated 2D array in contiguous memory
    //which means the whole thing can easily be copied to the GPU
    bitToXChecks = new int*[N]; 
    bitToXChecks[0] = new int[N*maxBitDegreeX]; 
    bitToZChecks = new int*[N];
    bitToZChecks[0] = new int[N*maxBitDegreeZ];
    xCheckToBits = new int*[M_X];
    xCheckToBits[0] = new int[M_X*maxCheckDegreeX];
    zCheckToBits = new int*[M_Z];
    zCheckToBits[0] = new int[M_Z*maxCheckDegreeZ];
    
    for (int i=1; i<N; ++i) 
    {
        bitToXChecks[i] = bitToXChecks[i-1] + maxBitDegreeX; 
        bitToZChecks[i] = bitToZChecks[i-1] + maxBitDegreeZ; 
    }
    for (int i=1; i<M_X; ++i) xCheckToBits[i] = xCheckToBits[i-1] + maxCheckDegreeX;
    for (int i=1; i<M_Z; ++i) zCheckToBits[i] = zCheckToBits[i-1] + maxCheckDegreeZ;

    buildBitToChecks(M_X, bitToXChecks, maxBitDegreeX, H_X);
    buildBitToChecks(M_Z, bitToZChecks, maxBitDegreeZ, H_Z);
    buildCheckToBits(M_X, xCheckToBits, maxCheckDegreeX, H_X);
    buildCheckToBits(M_Z, zCheckToBits, maxCheckDegreeZ, H_Z);
}

Code::~Code()
{
    delete[] bitToXChecks[0];
    delete[] bitToXChecks;
    delete[] bitToZChecks[0];
    delete[] bitToZChecks;
    delete[] xCheckToBits[0];
    delete[] xCheckToBits;
    delete[] zCheckToBits[0];
    delete[] zCheckToBits;
}
