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
        int maxVariableDegreeX;     //"variables" and "factors" are nodes in the factor graph
        int maxVariableDegreeZ;     //used in BP decoding. X variables come from the columns
        int maxFactorDegreeX;       //of the X parity check matrix and so actually represent
        int maxFactorDegreeZ;       //Z errors and X stab measurement errors (same for Z variables)
        int M_X;                
        int N_X;                    //These are the sizes of the X and Z parity check matrices
        int M_Z;                    
        int N_Z;                    

        //data arrays
        std::vector<std::vector<int>> H_X = {};
        std::vector<std::vector<int>> H_Z = {};
        int* variableDegreesX;      //Using pointers here because the GPU doesn't like vectors.//
        int* variableDegreesZ;      //As above these refer to variables and factors used in BP decoding.
        int** variableToFactorsX;   //There is always one variable per qubit (for physical Z/X errors)
        int** variableToFactorsZ;   //and one variable per X/Z stabiliser (for measurement errors)X/Z. 
        int* factorDegreesX;        //If there are no metachecks there will be one factor per X/Z stabiliser
        int* factorDegreesZ;        //and if there are metachecks there will also be one factor per X/Z metacheck//
        int** factorToVariablesX;   
        int** factorToVariablesZ;   
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
