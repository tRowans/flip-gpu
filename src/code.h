#ifndef CODE_H
#define CODE_H

#include<stdexcept>
#include<fstream>
#include<string>
#include<vector>

//Basically just a neat container for some data
class Code
{
    public:
        //params
        int N;
        int M_X;
        int M_Z;
        int maxBitDegreeX;
        int maxBitDegreeZ;
        int maxCheckDegreeX;
        int maxCheckDegreeZ;
        
        //data vector setup functions
        //these are only public so that they can be accessed in unit tests
        //apart from that they are not used outside of object construction
        void readParityCheckMatrix(std::string hFile, std::vector<std::vector<int>> &H);
        void findMaxBitDegree(std::vector<std::vector<int>> &H, int &maxBitDegree);
        void findMaxCheckDegree(std::vector<std::vector<int>> &mat, int &maxCheckDegree);
        void buildBitToChecks(int M, int** bitToChecks, int maxBitDegree, std::vector<std::vector<int>> &H);
        void buildCheckToBits(int M, int** checkToBits, int maxCheckDegree, std::vector<std::vector<int>> &H);

        //data arrays
        std::vector<std::vector<int>> H_X = {};
        std::vector<std::vector<int>> H_Z = {};
        int** bitToXChecks;
        int** bitToZChecks;
        int** xCheckToBits;
        int** zCheckToBits;

        Code(std::string codename);
        ~Code();
};

#endif
