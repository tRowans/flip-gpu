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
        int nQubits;
        int nChecksX;
        int nChecksZ;
        int nMetachecksX;
        int nMetachecksZ;
        int maxVariableDegreeX;     //"variables" and "factors" are node in the factor graph
        int maxVariableDegreeZ;     //used in BP decoding. When not using BP (or BP without metachecks)
        int maxFactorDegreeX;       //these will be equivalent to max bit degree and max check degree
        int maxFactorDegreeZ;       //in the X and Z tanner graphs
        int M_X;                
        int N_X;                    //These are the sizes of the X and Z parity check matrices
        int M_Z;                    //If we are not using BP then N_X = N_Z = nQubits, 
        int N_Z;                    //M_X = nChecksX and M_Z = nChecksZ

        //data arrays
        std::vector<std::vector<int>> H_X = {};
        std::vector<std::vector<int>> H_Z = {};
        int* variableDegreesX;      //Using pointers here because the GPU doesn't like vectors.//
        int* variableDegreesZ;      //As above these refer to variables and factors used in BP decoding.
        int** variableToFactorsX;   //When not using BP these will be arrays of bit/check degrees
        int** variableToFactorsZ;   //and maps from bits to connected checks and vice versa.
        int* factorDegreesX;        //When using BP the variable arrays will have extra entries 
        int* factorDegreesZ;        //for the measurement error variables, and when using BP  
        int** factorToVariablesX;   //with metachecks the factor arrays will have extra entries  
        int** factorToVariablesZ;   //for metacheck factors.//
        int** variableToPosX;       //These are maps from e.g. X variable v_i to an array [pos_1,pos_2,...,pos_n]
        int** variableToPosZ;       //where pos_j is such that for f_j in variableToFactorsX[v_i] = [f_1,f_2,...,f_n],
        int** factorToPosX;         //factorToVariablesX[f_j][pos_j] = v_i. In other words they let a variable find 
        int** factorToPosZ;         //its own position in the factor -> variables map of connected factors (and vice versa)//
        
        //data vector setup functions
        //these are only public so that they can be accessed in unit tests
        //apart from that they are not used outside of object construction
        void readParityCheckMatrix(std::string hFile, std::vector<std::vector<int>> &H);
        void getVariableDegrees(int M, int N, int* variableDegrees, int maxVariableDegree, std::vector<std::vector<int>> &H);
        void getFactorDegrees(int M, int N, int* factorDegrees, int maxFactorDegree, std::vector<std::vector<int>> &H);
        void buildVariableToFactors(int M, int N, int** variableToFactors, int maxVariableDegree, std::vector<std::vector<int>> &H);
        void buildFactorToVariables(int M, int N, int** factorToVariables, int maxFactorDegree, std::vector<std::vector<int>> &H);
        void buildNodeToPos(int nNodes, int** nodeToPos, int* nodeDegrees, int** nodeToNeighbours, 
                                    int maxNodeDegree, int* neighbourDegrees, int** neighbourToNodes);

        
        Code(std::string codename, int n);
        ~Code();
};

#endif
