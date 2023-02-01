#include "gtest/gtest.h"
#include "code.h"
#include "prepCode.h"

//------------------------------------------------------------

TEST(readParityCheckMatrixTest, surface)
{
    std::vector<std::vector<int>> H_X;
    std::vector<std::vector<int>> H_Z;
    H_X = {{1,1,1,1}};
    H_Z = {{1,0,1,0},{0,1,0,1}};
    EXPECT_EQ(H_X, fourqsurface.H_X);
    EXPECT_EQ(H_Z, fourqsurface.H_Z);
}
TEST(readParityCheckMatrixTest, colour2D)
{
    std::vector<std::vector<int>> H_X;
    std::vector<std::vector<int>> H_Z;
    H_X = {{1,1,0,1,1,0,0},
           {0,1,1,0,1,1,0},
           {0,0,0,1,1,1,1}};
    H_Z = {{1,1,0,1,1,0,0},
           {0,1,1,0,1,1,0},
           {0,0,0,1,1,1,1}};
    EXPECT_EQ(H_X, sevenqcolour.H_X);
    EXPECT_EQ(H_Z, sevenqcolour.H_Z);
}
TEST(readParityCheckMatrixTest, colour3D)
{
    std::vector<std::vector<int>> H_X;
    std::vector<std::vector<int>> H_Z;
    H_X = {{1,1,1,1,1,1,1,1}};
    H_Z = {{1,1,1,1,0,0,0,0,1,0,0,0,0,0},
           {1,1,0,0,1,1,0,0,0,1,0,0,0,0},
           {1,0,1,0,1,0,1,0,0,0,1,0,0,0},
           {0,1,0,1,0,1,0,1,0,0,0,1,0,0},
           {0,0,1,1,0,0,1,1,0,0,0,0,1,0},
           {0,0,0,0,1,1,1,1,0,0,0,0,0,1},
           {0,0,0,0,0,0,0,0,1,1,0,0,1,1},
           {0,0,0,0,0,0,0,0,1,0,1,1,0,1},
           {0,0,0,0,0,0,0,0,0,1,1,1,1,0}};
    EXPECT_EQ(eightq3Dcolour.H_X, H_X);
    EXPECT_EQ(eightq3Dcolour.H_Z, H_Z);
}

//------------------------------------------------------------

TEST(getVariableDegreesTest, surface)
{
    int variableDegreesX[4] = {1,1,1,1};
    int variableDegreesZ[4] = {1,1,1,1};
    EXPECT_EQ(fourqsurface.maxBitDegreeX, 1);
    EXPECT_EQ(fourqsurface.maxBitDegreeZ, 1);
    for (int i=0; i<4; ++i)
    {
        EXPECT_EQ(fourqsurface.variableDegreesX[i], variableDegreesX[i]);
        EXPECT_EQ(fourqsurface.variableDegreesZ[i], variableDegreesZ[i]);
    }
}
TEST(getVariableDegreesTest, colour2D)
{
    int variableDegreesX[7] = {1,2,1,2,3,2,1};
    int variableDegreesZ[7] = {1,2,1,2,3,2,1};
    EXPECT_EQ(sevenqcolour.maxVariableDegreeX, 3);
    EXPECT_EQ(sevenqcolour.maxVariableDegreeZ, 3);
    for (int i=0; i<7; ++i)
    {
        EXPECT_EQ(sevenqcolour.variableDegreesX[i], variableDegreesX[i]);
        EXPECT_EQ(sevenqcolour.variableDegreesZ[i], variableDegreesZ[i]);
    }
}
TEST(getVariableDegreesTest, colour3D)
{
    int variableDegreesX[14] = {3,3,3,3,3,3,3,3,3,3,3,3,3,3};
    int variableDegreesZ[8] = {1,1,1,1,1,1,1,1};
    EXPECT_EQ(eightq3Dcolour.maxVariableDegreeX, 3);
    EXPECT_EQ(eightq3Dcolour.maxVariableDegreeZ, 1);
    for (int i=0; i<14; ++i) EXPECT_EQ(eightq3Dcolour.variableDegreesX[i], variableDegreesX[i]);
    for (int i=0; i<8; ++i) EXPECT_EQ(eightq3Dcolour.variableDegreesZ[i], variableDegreesZ[i]);
}

//------------------------------------------------------------

TEST(getFactorDegreesTest, surface)
{
    int factorDegreesX[1] = {4};
    int factorDegreesZ[2] = {2,2};
    EXPECT_EQ(fourqsurface.maxFactorDegreeX, 4);
    EXPECT_EQ(fourqsurface.maxFactorDegreeZ, 2);
    EXPECT_EQ(fourqsurface.factorDegreesX[0], factorDegrees[0]);
    for (int i=0; i<2; ++i) EXPECT_EQ(fourqsurface.factorDegreesZ[i], factorDegreesZ[i]);
}
TEST(getFactorDegreesTest, colour2D)
{
    int factorDegreesX[3] = {4,4,4};
    int factorDegreesZ[3] = {4,4,4};
    EXPECT_EQ(sevenqcolour.maxFactorDegreeX, 4);
    EXPECT_EQ(sevenqcolour.maxFactorDegreeZ, 4);
    for (int i=0; i<3; ++i)
    {
        EXPECT_EQ(sevenqcolour.factorDegreesX[i], factorDegrees[i]);
        EXPECT_EQ(sevenqcolour.factorDegreesZ[i], factorDegrees[i]);
    }
}
TEST(getFactorDegreesTest, colour3D)
{
    int factorDegreesX[1] = {8};
    int factorDegreesZ[9] = {5,5,5,5,5,5,4,4,4,4};
    EXPECT_EQ(eightq3Dcolour.maxFactorDegreeX, 8);
    EXPECT_EQ(eightq3Dcolour.maxFactorDegreeZ, 5);
    EXPECT_EQ(eightq3Dcolour.factorDegreesX[0], factorDegreesX[0]);
    for (int i=0; i<9; ++i) EXPECT_EQ(eightq3Dcolour.factorDegreesZ[i], factorDegreesZ[i]);
}

//------------------------------------------------------------

TEST(buildVariableToFactors, surface)
{
    int variableToFactorsX[4][1] = {{0},{0},{0},{0}};
    int variableToFactorsZ[4][1] = {{0},{1},{0},{1}};
    for (int i=0; i<4; ++i)
    {
        EXPECT_EQ(fourqsurface.variableToFactorsX[i][0], variableToFactorsX[i][0]);
        EXPECT_EQ(fourqsurface.variableToFactorsZ[i][0], variableToFactorsZ[i][0]);
    }
}
TEST(buildVariableToFactors, colour2D)
{
    int variableToFactorsX[7][3] = {{0,-1,-1},
                                    {0,1,-1},
                                    {1,-1,-1},
                                    {0,2,-1},
                                    {0,1,2},
                                    {1,2,-1},
                                    {2,-1,-1}};
    int variableToFactorsZ[7][3] = {{0,-1,-1},
                                    {0,1,-1},
                                    {1,-1,-1},
                                    {0,2,-1},
                                    {0,1,2},
                                    {1,2,-1},
                                    {2,-1,-1}};
    for (int i=0; i<7; ++i)
    {
        for (int j=0; j<3; ++j)
        {
            EXPECT_EQ(sevenqcolour.variableToFactorsX[i][j], variableToFactorsX[i][j]);
            EXPECT_EQ(sevenqcolour.variableToFactorsZ[i][j], variableToFactorsZ[i][j]);
        }
    }
}
TEST(buildVariableToFactors, colour3D)
{
    int variableToFactorsX[14][3] = {{0,1,2},
                                     {0,1,3},
                                     {0,2,4},
                                     {0,3,4},
                                     {1,2,5},
                                     {1,3,5},
                                     {2,4,5},
                                     {3,4,5},
                                     {0,6,7},
                                     {1,6,8},
                                     {2,7,8},
                                     {3,7,8},
                                     {4,6,8},
                                     {5,6,7}}
    int variableToFactorsZ[8][1] = {{0},{0},{0},{0},{0},{0},{0},{0}};
    for (int i=0; i<14; ++i)
    {
        for (int j=0; j<3; ++j) EXPECT_EQ(eightq3Dcolour.variableToFactorsX[i][j], variableToFactorsX[i][j]);
    }
    for (int i=0; i<8; ++i) EXPECT_EQ(eightq3Dcolour.variableToFactorsZ[i][0], variableToFactorsZ[i][0]);
}
        
//------------------------------------------------------------

TEST(buildFactorToVariables, surface)
{
    int factorToVariablesX[1][4] = {{0,1,2,3}};
    int factorToVariablesZ[2][2] = {{0,2},{1,3}};
    for (int i=0; i<4; ++i) EXPECT_EQ(fourqsurface.factorToVariablesX[0][i], factorToVariablesX[0][i]);
    for (int i=0; i<2; ++i)
    {
        for (int j=0; j<2; ++j) EXPECT_EQ(fourqsurface.factorToVariablesZ[i][j], factorToVariablesZ[i][j]);
    }
}
TEST(buildFactorToVariables, colour2D)
{
    int factorToVariablesX[3][4] = {{0,1,3,4},
                                    {1,2,4,5},
                                    {3,4,5,6}};
    int factorToVariablesZ[3][4] = {{0,1,3,4},
                                    {1,2,4,5},
                                    {3,4,5,6}};
    for (int i=0; i<3; ++i)
    {
        for (int j=0; j<4; ++j)
        {
            EXPECT_EQ(sevenqcolour.factorToVariablesX[i][j], factorToVariablesX[i][j]);
            EXPECT_EQ(sevenqcolour.factorToVariablesZ[i][j], factorToVariablesZ[i][j]);
        }
    }
}
TEST(buildFactorToVariables, colour3D)
{
    int factorToVariablesX[1][8] = {{0,1,2,3,4,5,6,7}};
    int factorToVariablesZ[9][5] = {{0,1,2,3,8},
                                    {0,1,4,5,9},
                                    {0,2,4,6,10},
                                    {1,3,5,7,11},
                                    {2,3,6,7,12},
                                    {4,5,6,7,13},
                                    {8,9,12,13,-1},
                                    {8,10,11,13,-1},
                                    {9,10,11,12,-1}};
    for (int i=0; i<8; ++i) EXPECT_EQ(eightq3Dcolour.factorToVariablesX[0][i], factorToVariablesX[0][i]);
    for (int i=0; i<9; ++i)
    {
        for (int j=0; j<5; ++j) EXPECT_EQ(eightq3Dcolour.factorToVariablesZ[i][j], factorToVariablesZ[i][j]);
    }
}

//------------------------------------------------------------

TEST(buildNodeToPosTest, surface)
{
    int variableToPosX[4][1] = {{0},{1},{2},{3}};
    int variableToPosZ[4][1] = {{0},{0},{1},{1}};
    int factorToPosX[1][4] = {0,0,0,0};
    int factorToPosZ[2][2] = {{0,0},{0,0}};
    for (int i=0; i<4; ++i)
    {
        EXPECT_EQ(fourqsurface.variableToPosX[i][0], variableToPosX[i][0]);
        EXPECT_EQ(fourqsurface.variableToPosZ[i][0], variableToPosZ[i][0]);
        EXPECT_EQ(fourqsurface.factorToPosX[0][i], factorToPosX[0][i]);
    }
    for (int i=0; i<2; ++i)
    {
        for (int j=0; j<2; ++j) EXPECT_EQ(fourqsurface.factorToPosZ[i][j], factorToPosZ[i][j]);
    }
}
TEST(buildNodeToPosTest, colour2D)
{
    int variableToPosX[7][3] = {{0,-1,-1},
                                {1,0,-1},
                                {1,-1,-1},
                                {2,0,-1},
                                {3,2,1},
                                {3,2,-1},
                                {3,-1,-1}};
    int variableToPosZ[7][3] = {{0,-1,-1},
                                {1,0,-1},
                                {1,-1,-1},
                                {2,0,-1},
                                {3,2,1},
                                {3,2,-1},
                                {3,-1,-1}};
    int factorToPosX[3][4] = {{0,0,0,0},
                              {1,0,1,0},
                              {1,2,1,0}};
    int factorToPosZ[3][4] = {{0,0,0,0},
                              {1,0,1,0},
                              {1,2,1,0}};
    for (int i=0; i<7; ++i)
    {
        for (int j=0; j<3; ++j)
        {
            EXPECT_EQ(sevenqcolour.variableToPosX[i][j], variableToPosX[i][j]);
            EXPECT_EQ(sevenqcolour.variableToPosZ[i][j], variableToPosZ[i][j]);
        }
    }
    for (int i=0; i<3; ++i)
    {
        for (int j=0; j<4; ++j)
        {
            EXPECT_EQ(sevenqcolour.factorToPosX[i][j], factorToPosX[i][j]);
            EXPECT_EQ(sevenqcolour.factorToPosZ[i][j], factorToPosZ[i][j]);
        }
    }
}
TEST(buildNodeToPosTest, colour3D)
{
    int variableToPosX[14][3] = {{0,0,0},
                                 {1,1,0},
                                 {2,1,0},
                                 {3,1,1},
                                 {2,2,0},
                                 {3,2,1},
                                 {3,2,2},
                                 {3,3,3},
                                 {4,0,0},
                                 {4,1,0},
                                 {4,1,1},
                                 {4,2,2},
                                 {4,2,3},
                                 {4,3,3}};
    int variableToPosZ[8][1] = {{0},{1},{2},{3},{4},{5},{6},{7}};
    int factorToPosX[1][8] = {{0,0,0,0,0,0,0,0}};
    int factorToPosZ[9][5] = {{0,0,0,0,0},
                              {1,1,0,0,0},
                              {2,1,1,0,0},
                              {2,1,1,0,0},
                              {2,2,1,1,0},
                              {2,2,2,2,0},
                              {1,1,1,1,-1},
                              {2,1,1,2,-1},
                              {2,2,2,2,-1}};
    for (int i=0; i<14; ++i)
    {
        for (int j=0; j<3; ++j) EXPECT_EQ(eightq3Dcolour.variableToPosX[i][j], variableToPos[i][j]);
    }
    for (int i=0; i<8; ++i)
    {
        EXPECT_EQ(eightq3Dcolour.variableToPosZ[i][0], variableToPosZ[i][0]);
        EXPECT_EQ(eightq3Dcolour.factorToPosX[0][i], factorToPosX[0][i]);
    }
    for (int i=0; i<9; ++i)
    {
        for (int j=0; j<5; ++j) EXPECT_EQ(eightq3Dcolour.factorToPosZ[i][j], factorToPosZ[i][j]);
    }
}
