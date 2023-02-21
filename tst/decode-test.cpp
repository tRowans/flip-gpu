#include "gtest/gtest.h"
#include "decode_wrappers.cuh"
#include "prepCode.h"
#include<random>

//------------------------------------------------------------

TEST(wipeArrayTest, CorrectOutput)
{
    int arr[100];
    for (int i=0; i<100; ++i)
    {
        arr[i] = 1;
    }
    wipeArrayWrap(100, arr);
    for(int i=0; i<100; ++i)
    {
        EXPECT_EQ(arr[i], 0);
    }
}

//------------------------------------------------------------

TEST(depolErrorsTest, NonTrivialOutput)
{
    int variablesX[100] = {};
    int variablesZ[100] = {};
    std::random_device rd{};
    depolErrorsWrap(100, 100, 100, rd(), variablesX, variablesZ, 0.5);
    int totalQX = 0;
    int totalQZ = 0;
    for (int i=0; i<100; ++i)
    {
        totalQX += variablesX[i];
        totalQZ += variablesZ[i];
    }
    EXPECT_NE(totalQX, 0);
    EXPECT_NE(totalQZ, 0);
}
TEST(depolErrorsTest, ProbabilityOne)
{  
    int variablesX[100] = {};
    int variablesZ[100] = {};
    std::random_device rd{};
    depolErrorsWrap(100, 100, 100, rd(), variablesX, variablesZ, 1);
    int totalQX = 0;
    int totalQZ = 0;
    int totalEither = 0;
    for (int i=0; i<100; ++i)
    {
        totalQX += variablesX[i];
        totalQZ += variablesZ[i];
        totalEither += (variablesX[i] || variablesZ[i]);
    }
    EXPECT_NE(totalQX, 0);
    EXPECT_NE(totalQZ, 0);
    EXPECT_EQ(totalEither, 100);
}
TEST(depolErrorsTest, MoreVariablesThanQubits)
{
    int variablesX[200] = {};
    int variablesZ[200] = {};
    std::random_device rd{};
    depolErrorsWrap(100, 100, 100, rd(), variablesX, variablesZ, 1);
    int totalQX = 0;
    int totalQZ = 0;
    int totalEither = 0;
    int totalSZ = 0;
    int totalSX = 0;
    for (int i=0; i<200; ++i)
    {
        totalQX += variablesX[i];
        totalQZ += variablesZ[i];
        totalEither += (variablesX[i] || variablesZ[i]);
        if (i >= 100)
        {
            totalSZ += variablesX[i];
            totalSX += variablesZ[i];
        }
    }
    EXPECT_NE(totalQX, 0);
    EXPECT_NE(totalQZ, 0);
    EXPECT_EQ(totalEither, 100);
    //no change to measurement errors
    EXPECT_EQ(totalSZ, 0);
    EXPECT_EQ(totalSX, 0);
}

//------------------------------------------------------------

TEST(measErrorsTest, NonTrivialOutput)
{
    int variables[200] = {};
    std::random_device rd{};
    measErrorsWrap(100, 100, rd(), variables, 0.5);
    int totalQ = 0;
    int totalS = 0;
    for (int i=0; i<100; ++i)
    {
        totalQ += variables[i];
        totalS += variables[100+i];
    }
    EXPECT_EQ(totalQ, 0);   //no change to qubit errors
    EXPECT_NE(totalS, 0);
}
TEST(measErrorsTest, ProbabilityOne)
{
    int variables[200] = {};
    std::random_device rd{};
    measErrorsWrap(100, 100, rd(), variables, 1);
    int totalQ = 0;
    int totalS = 0;
    for (int i=0; i<100; ++i)
    {
        totalQ += variables[i];
        totalS += variables[100+i];
    }
    EXPECT_EQ(totalQ, 0);   //no change to qubit errors
    EXPECT_EQ(totalS, 100);
}
TEST(measErrorsTest, CheckOverwritesPastErrors)
{
    int variables[2] = {0,1};
    std::random_device rd{};
    measErrorsWrap(1, 1, rd(), variables, 0);
    EXPECT_EQ(variables[0], 0);
    EXPECT_EQ(variables[1], 0);
}

//------------------------------------------------------------

//qubits should never change from this function
TEST(calculateSyndromeTest, surface)
{
    int variablesX[6] = {1,0,0,0,0,0};
    int variablesZ[5] = {1,0,0,0,0};
    int factorsZ[2] = {};
    int factorsX[1] = {};
    int variablesXExpected[6] = {1,0,0,0,0,0};
    int variablesZExpected[5] = {1,0,0,0,0};
    int factorsZExpected[2] = {1,0};
    int factorsXExpected[1] = {1};
    calculateSyndromeWrap(6, 2, variablesX, factorsZ, fourqsurface.factorToVariablesZ, 
                            fourqsurface.factorDegreesZ, fourqsurface.maxFactorDegreeZ);
    calculateSyndromeWrap(5, 1, variablesZ, factorsX, fourqsurface.factorToVariablesX, 
                            fourqsurface.factorDegreesX, fourqsurface.maxFactorDegreeX);
    for (int i=0; i<6; ++i) EXPECT_EQ(variablesX[i], variablesXExpected[i]);
    for (int i=0; i<5; ++i) EXPECT_EQ(variablesZ[i], variablesZExpected[i]);
    EXPECT_EQ(factorsZ[0], factorsZExpected[0]);
    EXPECT_EQ(factorsZ[1], factorsZExpected[1]);
    EXPECT_EQ(factorsX[0], factorsXExpected[0]);
}
TEST(calculateSyndromeTest, colour2D)
{
    int variablesX[10] = {0,0,0,0,1,0,0,0,0,0};
    int variablesZ[10] = {0,1,0,0,0,0,0,0,0,0};
    int factorsZ[3] = {};
    int factorsX[3] = {};
    int variablesXExpected[10] = {0,0,0,0,1,0,0,0,0,0};
    int variablesZExpected[10] = {0,1,0,0,0,0,0,0,0,0};
    int factorsZExpected[3] = {1,1,1};
    int factorsXExpected[3] = {1,1,0};
    calculateSyndromeWrap(10, 3, variablesX, factorsZ, sevenqcolour.factorToVariablesZ, 
                            sevenqcolour.factorDegreesZ, sevenqcolour.maxFactorDegreeZ);
    calculateSyndromeWrap(10, 3, variablesZ, factorsX, sevenqcolour.factorToVariablesX, 
                            sevenqcolour.factorDegreesX, sevenqcolour.maxFactorDegreeX);
    for (int i=0; i<10; ++i)
    {
        EXPECT_EQ(variablesX[i], variablesXExpected[i]);
        EXPECT_EQ(variablesZ[i], variablesZExpected[i]);
    }
    for (int i=0; i<3; ++i)
    {
        EXPECT_EQ(factorsZ[i], factorsZExpected[i]);
        EXPECT_EQ(factorsX[i], factorsXExpected[i]);
    }
}
TEST(calculateSyndromeTest, colour2DMeasError)
{
    int variablesX[10] = {0,0,0,0,1,0,0,1,0,0};
    int variablesZ[10] = {0,1,0,0,0,0,0,0,0,1};
    int factorsZ[3] = {};
    int factorsX[3] = {};
    int variablesXExpected[10] = {0,0,0,0,1,0,0,1,0,0};
    int variablesZExpected[10] = {0,1,0,0,0,0,0,0,0,1};
    int factorsZExpected[3] = {0,1,1};
    int factorsXExpected[3] = {1,1,1};
    calculateSyndromeWrap(10, 3, variablesX, factorsZ, sevenqcolour.factorToVariablesZ, 
                            sevenqcolour.factorDegreesZ, sevenqcolour.maxFactorDegreeZ);
    calculateSyndromeWrap(10, 3, variablesZ, factorsX, sevenqcolour.factorToVariablesX,
                            sevenqcolour.factorDegreesX, sevenqcolour.maxFactorDegreeX);
    for (int i=0; i<10; ++i)
    {
        EXPECT_EQ(variablesX[i], variablesXExpected[i]);
        EXPECT_EQ(variablesZ[i], variablesZExpected[i]);
    }
    for (int i=0; i<3; ++i)
    {
        EXPECT_EQ(factorsZ[i], factorsZExpected[i]);
        EXPECT_EQ(factorsX[i], factorsXExpected[i]);
    }
}
TEST(calculateSyndromeTest, colour3D)
{
    int variablesX[14] = {0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    int variablesZ[9] = {1,0,0,0,0,0,0,0,0};
    int factorsZ[9] = {};
    int factorsX[1] = {};
    int variablesXExpected[14] = {0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    int variablesZExpected[9] = {1,0,0,0,0,0,0,0,0};
    int factorsZExpected[9] = {1,0,1,0,1,0,0,0,0};
    int factorsXExpected[1] = {1};
    calculateSyndromeWrap(14, 9, variablesX, factorsZ, eightq3Dcolour.factorToVariablesZ, 
                            eightq3Dcolour.factorDegreesZ, eightq3Dcolour.maxFactorDegreeZ);
    calculateSyndromeWrap(9, 1, variablesZ, factorsX, eightq3Dcolour.factorToVariablesX, 
                            eightq3Dcolour.factorDegreesX, eightq3Dcolour.maxFactorDegreeX);
    for (int i=0; i<14; ++i) EXPECT_EQ(variablesX[i], variablesXExpected[i]);
    for (int i=0; i<9; ++i)
    {
        EXPECT_EQ(variablesZ[i], variablesZExpected[i]);
        EXPECT_EQ(factorsZ[i], factorsZExpected[i]);
    }
    EXPECT_EQ(factorsX[0], factorsXExpected[0]);
}
TEST(calculateSyndromeTest, colour3DMeasError)
{
    int variablesX[14] = {0,0,1,0,0,0,0,0,0,0,0,0,0,1};
    int variablesZ[9] = {1,0,0,0,0,0,0,0,0};
    int factorsZ[9] = {};
    int factorsX[1] = {};
    int variablesXExpected[14] = {0,0,1,0,0,0,0,0,0,0,0,0,0,1};
    int variablesZExpected[9] = {1,0,0,0,0,0,0,0,0};
    int factorsZExpected[9] = {1,0,1,0,1,1,1,1,0};
    int factorsXExpected[1] = {1};
    calculateSyndromeWrap(14, 9, variablesX, factorsZ, eightq3Dcolour.factorToVariablesZ, 
                            eightq3Dcolour.factorDegreesZ, eightq3Dcolour.maxFactorDegreeZ);
    calculateSyndromeWrap(9, 1, variablesZ, factorsX, eightq3Dcolour.factorToVariablesX, 
                            eightq3Dcolour.factorDegreesX, eightq3Dcolour.maxFactorDegreeX);
    for (int i=0; i<14; ++i) EXPECT_EQ(variablesX[i], variablesXExpected[i]);
    for (int i=0; i<9; ++i)
    {
        EXPECT_EQ(variablesZ[i], variablesZExpected[i]);
        EXPECT_EQ(factorsZ[i], factorsZExpected[i]);
    }
    EXPECT_EQ(factorsX[0], factorsXExpected[0]);
}

//------------------------------------------------------------

TEST(flipTest, surface)
{
    int variablesX[6] = {1,0,0,0,0,0};
    int variablesZ[5] = {1,0,0,0,0};
    int factorsZ[2] = {1,0};
    int factorsX[1] = {1};
    int variablesXExpected[6] = {0,0,1,0,0,0};
    int variablesZExpected[5] = {0,1,1,1,0};
    int factorsZExpected[2] = {1,0};
    int factorsXExpected[1] = {1};
    flipWrap(6, 2, 4, 2, variablesX, factorsZ, fourqsurface.variableToFactorsX, 
                fourqsurface.variableDegreesX, fourqsurface.maxVariableDegreeX);
    flipWrap(5, 1, 4, 1, variablesZ, factorsX, fourqsurface.variableToFactorsZ, 
                fourqsurface.variableDegreesZ, fourqsurface.maxVariableDegreeZ);
    for (int i=0; i<6; ++i) EXPECT_EQ(variablesX[i], variablesXExpected[i]);
    for (int i=0; i<5; ++i) EXPECT_EQ(variablesZ[i], variablesZExpected[i]);
    EXPECT_EQ(factorsZ[0], factorsZExpected[0]);
    EXPECT_EQ(factorsZ[1], factorsZExpected[1]);
    EXPECT_EQ(factorsX[0], factorsXExpected[0]);
}
TEST(flipTest, colour2D)
{
    int variablesX[10] = {0,0,0,0,1,0,0,0,0,0};
    int variablesZ[10] = {0,1,0,0,0,0,0,0,0,0};
    int factorsZ[3] = {1,1,1};
    int factorsX[3] = {1,1,0};
    int variablesXExpected[10] = {1,1,1,1,0,1,1,0,0,0};
    int variablesZExpected[10] = {1,0,1,0,1,0,0,0,0,0};
    int factorsZExpected[3] = {1,1,1};
    int factorsXExpected[3] = {0,0,1};
    flipWrap(10, 3, 7, 3, variablesX, factorsZ, sevenqcolour.variableToFactorsX, 
                sevenqcolour.variableDegreesX, sevenqcolour.maxVariableDegreeX);
    flipWrap(10, 3, 7, 3, variablesZ, factorsX, sevenqcolour.variableToFactorsZ, 
                sevenqcolour.variableDegreesZ, sevenqcolour.maxVariableDegreeZ);
    for (int i=0; i<10; ++i)
    {
        EXPECT_EQ(variablesX[i], variablesXExpected[i]);
        EXPECT_EQ(variablesZ[i], variablesZExpected[i]);
    }
    for (int i=0; i<3; ++i)
    {
        EXPECT_EQ(factorsZ[i], factorsZExpected[i]);
        EXPECT_EQ(factorsX[i], factorsXExpected[i]);
    }
}
TEST(flipTest, colour2DMeasError)
{
    int variablesX[10] = {0,0,0,0,1,0,0,0,0,0};
    int variablesZ[10] = {0,1,0,0,0,0,0,1,0,0};
    int factorsZ[3] = {1,1,1};
    int factorsX[3] = {0,1,0};
    int variablesXExpected[10] = {1,1,1,1,0,1,1,0,0,0};
    int variablesZExpected[10] = {0,1,1,0,0,0,0,1,0,0};
    int factorsZExpected[3] = {1,1,1};
    int factorsXExpected[3] = {0,0,0};
    flipWrap(10, 3, 7, 3, variablesX, factorsZ, sevenqcolour.variableToFactorsX, 
                sevenqcolour.variableDegreesX, sevenqcolour.maxVariableDegreeX);
    flipWrap(10, 3, 7, 3, variablesZ, factorsX, sevenqcolour.variableToFactorsZ, 
                sevenqcolour.variableDegreesZ, sevenqcolour.maxVariableDegreeZ);
    for (int i=0; i<10; ++i)
    {
        EXPECT_EQ(variablesX[i], variablesXExpected[i]);
        EXPECT_EQ(variablesZ[i], variablesZExpected[i]);
    }
    for (int i=0; i<3; ++i)
    {
        EXPECT_EQ(factorsZ[i], factorsZExpected[i]);
        EXPECT_EQ(factorsX[i], factorsXExpected[i]);
    }
}
TEST(flipTest, colour3D)
{
    int variablesX[14] = {0,0,1,0,0,0,0,0,0,0,0,0,0,0};
    int variablesZ[9] = {1,0,0,0,0,0,0,0,0};
    int factorsZ[9] = {1,0,1,0,1,0,0,0,0};
    int factorsX[1] = {1};
    int variablesXExpected[14] = {1,0,0,1,0,0,1,0,0,0,0,0,0,0};
    int variablesZExpected[9] = {0,1,1,1,1,1,1,1,0};
    int factorsZExpected[9] = {0,1,0,1,0,1,0,0,0};
    int factorsXExpected[1] = {1};
    flipWrap(14, 9, 8, 6, variablesX, factorsZ, eightq3Dcolour.variableToFactorsX, 
                 eightq3Dcolour.variableDegreesX, eightq3Dcolour.maxVariableDegreeX);
    flipWrap(9, 1, 8, 1, variablesZ, factorsX, eightq3Dcolour.variableToFactorsZ,
                 eightq3Dcolour.variableDegreesZ, eightq3Dcolour.maxVariableDegreeX);
    for (int i=0; i<14; ++i)
    {
        EXPECT_EQ(variablesX[i], variablesXExpected[i]);
        EXPECT_EQ(variablesZ[i], variablesZExpected[i]);
    }
    for (int i=0; i<9; ++i) EXPECT_EQ(factorsZ[i], factorsZExpected[i]);
    EXPECT_EQ(factorsX[0], factorsXExpected[0]);
}
TEST(flipTest, colour3DMeasError)
{
    int variablesX[14] = {0,0,1,0,0,0,0,0,0,0,0,0,0,1};
    int variablesZ[9] = {1,0,0,0,0,0,0,0,0};
    int factorsZ[9] = {1,0,1,0,1,1,1,1,0};
    int factorsX[1] = {1};
    int variablesXExpected[14] = {1,0,0,1,1,0,1,1,0,0,0,0,0,1};
    int variablesZExpected[9] = {0,1,1,1,1,1,1,1,0};
    int factorsZExpected[9] = {0,0,1,0,1,0,1,1,0};
    int factorsXExpected[1] = {1};
    flipWrap(14, 9, 8, 6, variablesX, factorsZ, eightq3Dcolour.variableToFactorsX, 
                 eightq3Dcolour.variableDegreesX, eightq3Dcolour.maxVariableDegreeX);
    flipWrap(9, 1, 8, 1, variablesZ, factorsX, eightq3Dcolour.variableToFactorsZ,
                 eightq3Dcolour.variableDegreesZ, eightq3Dcolour.maxVariableDegreeX);
    for (int i=0; i<14; ++i)
    {
        EXPECT_EQ(variablesX[i], variablesXExpected[i]);
        EXPECT_EQ(variablesZ[i], variablesZExpected[i]);
    }
    for (int i=0; i<9; ++i) EXPECT_EQ(factorsZ[i], factorsZExpected[i]);
    EXPECT_EQ(factorsX[0], factorsXExpected[0]);
}

//------------------------------------------------------------

TEST(pflipTest, surface)
{
    //This one identical to flip because each qubit is part of only one stab of each type
    int variablesX[6] = {1,0,0,0,0,0};
    int variablesZ[5] = {1,0,0,0,0};
    int factorsZ[2] = {1,0};
    int factorsX[1] = {1};
    int variablesXExpected[6] = {0,0,1,0,0,0};
    int variablesZExpected[5] = {0,1,1,1,0};
    int factorsZExpected[2] = {1,0};
    int factorsXExpected[1] = {1};
    std::random_device rd{};
    pflipWrap(6, 2, 4, 2, rd(), variablesX, factorsZ, fourqsurface.variableToFactorsX, 
                fourqsurface.variableDegreesX, fourqsurface.maxVariableDegreeX);
    pflipWrap(5, 1, 4, 1, rd(), variablesZ, factorsX, fourqsurface.variableToFactorsZ, 
                fourqsurface.variableDegreesZ, fourqsurface.maxVariableDegreeZ);
    for (int i=0; i<6; ++i) EXPECT_EQ(variablesX[i], variablesXExpected[i]);
    for (int i=0; i<5; ++i) EXPECT_EQ(variablesZ[i], variablesZExpected[i]);
    EXPECT_EQ(factorsZ[0], factorsZExpected[0]);
    EXPECT_EQ(factorsZ[1], factorsZExpected[1]);
    EXPECT_EQ(factorsX[0], factorsXExpected[0]);
}
TEST(pflipTest, colour2D)
{
    int variablesX[10] = {0,0,0,0,1,0,0,0,0,0};
    int variablesZ[10] = {0,1,0,0,0,0,0,0,0,0};
    int factorsZ[3] = {1,1,1};
    int factorsX[3] = {1,1,0};
    int variablesXExpected[10] = {1,1,1,1,0,1,1,0,0,0};
    int variablesZExpected1[10] = {1,0,1,0,1,0,0,0,0,0};
    int variablesZExpected2[10] = {1,0,1,1,1,0,0,0,0,0};
    int variablesZExpected3[10] = {1,0,1,0,1,1,0,0,0,0};
    int variablesZExpected4[10] = {1,0,1,1,1,1,0,0,0,0};
    int factorsZExpected[3] = {1,1,1};
    int factorsXExpected1[3] = {0,0,1};
    int factorsXExpected2[3] = {1,0,0};
    int factorsXExpected3[3] = {0,1,0};
    int factorsXExpected4[3] = {1,1,1};
    std::random_device rd{};
    pflipWrap(10, 3, 7, 3, rd(), variablesX, factorsZ, sevenqcolour.variableToFactorsX, 
                sevenqcolour.variableDegreesX, sevenqcolour.maxVariableDegreeX);
    pflipWrap(10, 3, 7, 3, rd(), variablesZ, factorsX, sevenqcolour.variableToFactorsZ, 
                sevenqcolour.variableDegreesZ, sevenqcolour.maxVariableDegreeZ);
    int matches[4] = {1,1,1,1};
    for (int i=0; i<10; ++i)
    {
        EXPECT_EQ(variablesX[i], variablesXExpected[i]);
        if (variablesZ[i] != variablesZExpected1[i]) matches[0] = 0;
        if (variablesZ[i] != variablesZExpected2[i]) matches[1] = 0;
        if (variablesZ[i] != variablesZExpected3[i]) matches[2] = 0;
        if (variablesZ[i] != variablesZExpected4[i]) matches[3] = 0;
    }
    int match = matches[0] + matches[1] + matches[2] + matches[3];
    EXPECT_EQ(match, 1);
    for (int i=0; i<4; ++i) matches[i] = 1;
    for (int i=0; i<3; ++i)
    {
        EXPECT_EQ(factorsZ[i], factorsZExpected[i]);
        if (factorsX[i] != factorsXExpected1[i]) matches[0] = 0;
        if (factorsX[i] != factorsXExpected2[i]) matches[1] = 0;
        if (factorsX[i] != factorsXExpected3[i]) matches[2] = 0;
        if (factorsX[i] != factorsXExpected4[i]) matches[3] = 0;
    }
    match = matches[0] + matches[1] + matches[2] + matches[3];
    EXPECT_EQ(match,1);
}
TEST(pflipTest, colour2DMeasError)
{
    int variablesX[10] = {0,0,0,0,1,0,0,0,0,0};
    int variablesZ[10] = {0,1,0,0,0,0,0,1,0,0};
    int factorsZ[3] = {1,1,1};
    int factorsX[3] = {0,1,0};
    int variablesXExpected[10] = {1,1,1,1,0,1,1,0,0,0};
    int variablesZExpected1[10] = {0,1,1,0,0,0,0,1,0,0};
    int variablesZExpected2[10] = {0,0,1,0,0,0,0,1,0,0};
    int variablesZExpected3[10] = {0,1,1,0,0,1,0,1,0,0};
    int variablesZExpected4[10] = {0,0,1,0,0,1,0,1,0,0};
    int factorsZExpected[3] = {1,1,1};
    int factorsXExpected1[3] = {0,0,0};
    int factorsXExpected2[3] = {1,1,0};
    int factorsXExpected3[3] = {0,1,0};
    int factorsXExpected4[3] = {1,0,1};
    std::random_device rd{};
    pflipWrap(10, 3, 7, 3, rd(), variablesX, factorsZ, sevenqcolour.variableToFactorsX, 
                sevenqcolour.variableDegreesX, sevenqcolour.maxVariableDegreeX);
    pflipWrap(10, 3, 7, 3, rd(), variablesZ, factorsX, sevenqcolour.variableToFactorsZ, 
                sevenqcolour.variableDegreesZ, sevenqcolour.maxVariableDegreeZ);
    int matches[4] = {1,1,1,1};
    for (int i=0; i<10; ++i)
    {
        EXPECT_EQ(variablesX[i], variablesXExpected[i]);
        if (variablesZ[i] != variablesZExpected1[i]) matches[0] = 0;
        if (variablesZ[i] != variablesZExpected2[i]) matches[1] = 0;
        if (variablesZ[i] != variablesZExpected3[i]) matches[2] = 0;
        if (variablesZ[i] != variablesZExpected4[i]) matches[3] = 0;
    }
    int match = matches[0] + matches[1] + matches[2] + matches[3];
    EXPECT_EQ(match, 1);
    for (int i=0; i<4; ++i) matches[i] = 1;
    for (int i=0; i<3; ++i)
    {
        EXPECT_EQ(factorsZ[i], factorsZExpected[i]);
        if (factorsX[i] != factorsXExpected1[i]) matches[0] = 0;
        if (factorsX[i] != factorsXExpected2[i]) matches[1] = 0;
        if (factorsX[i] != factorsXExpected3[i]) matches[2] = 0;
        if (factorsX[i] != factorsXExpected4[i]) matches[3] = 0;
    }
    match = matches[0] + matches[1] + matches[2] + matches[3];
    EXPECT_EQ(match,1);
}
//All qubits in colour3D are part of 3 Z checks so pflip is the same as flip

//------------------------------------------------------------

TEST(initVariableMessagesTest, surface)
{
    double variableMessagesX[2*3] = {};
    double variableMessagesZ[1*5] = {};
    double llrp0 = log10(0.99/0.01);
    double llrq0 = log10(0.98/0.02);
    double variableMessagesXExpected[2*3] = {llrp0, llrp0, llrq0, llrp0, llrp0, llrq0};
    double variableMessagesZExpected[1*5] = {llrp0, llrp0, llrp0, llrp0, llrq0};
    initVariableMessagesWrap(2, 2, variableMessagesX, fourqsurface.factorDegreesZ, fourqsurface.maxFactorDegreeZ, llrp0, llrq0);
    initVariableMessagesWrap(1, 1, variableMessagesZ, fourqsurface.factorDegreesX, fourqsurface.maxFactorDegreeX, llrp0, llrq0);
    for (int i=0; i<6; ++i) EXPECT_DOUBLE_EQ(variableMessagesX[i], variableMessagesXExpected[i]);
    for (int i=0; i<5; ++i) EXPECT_DOUBLE_EQ(variableMessagesZ[i], variableMessagesZExpected[i]);
}
//Not sure there's any point in testing 2D colour here because it doesn't test anything not tested by surface or 3D colour
TEST(initVariableMessagesTest, colour3D)
{
    double variableMessagesX[9*5] = {};
    double variableMessagesZ[1*9] = {};
    double llrp0 = log10(0.99/0.01);
    double llrq0 = log10(0.98/0.02);
    double variableMessagesXExpected[9*5] = {llrp0,llrp0,llrp0,llrp0,llrq0,
                                             llrp0,llrp0,llrp0,llrp0,llrq0,
                                             llrp0,llrp0,llrp0,llrp0,llrq0,
                                             llrp0,llrp0,llrp0,llrp0,llrq0,
                                             llrp0,llrp0,llrp0,llrp0,llrq0,
                                             llrp0,llrp0,llrp0,llrp0,llrq0,
                                             llrq0,llrq0,llrq0,llrq0,0,
                                             llrq0,llrq0,llrq0,llrq0,0,
                                             llrq0,llrq0,llrq0,llrq0,0};
    double variableMessagesZExpected[1*9] = {llrp0, llrp0, llrp0, llrp0, llrp0, llrp0, llrp0, llrp0, llrq0};
    initVariableMessagesWrap(9, 6, variableMessagesX, eightq3Dcolour.factorDegreesZ, eightq3Dcolour.maxFactorDegreeZ, llrp0, llrq0);
    initVariableMessagesWrap(1, 1, variableMessagesZ, eightq3Dcolour.factorDegreesX, eightq3Dcolour.maxFactorDegreeX, llrp0, llrq0);
    for (int i=0; i<9*5; ++i) EXPECT_DOUBLE_EQ(variableMessagesX[i], variableMessagesXExpected[i]);
    for (int i=0; i<1*9; ++i) EXPECT_DOUBLE_EQ(variableMessagesZ[i], variableMessagesZExpected[i]);
}

//------------------------------------------------------------

//Think only 3D colour is useful to test here
TEST(updateFactorMessagesTanhTest, CorrectOutput)
{
    int factorsZ[9] = {1,0,1,0,1,1,1,1,0};
    double llrp0 = log10(0.99/0.01);
    double llrq0 = log10(0.98/0.02);
    double variableMessagesX[9*5] = {llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrq0,llrq0,llrq0,llrq0,0,
                                     llrq0,llrq0,llrq0,llrq0,0,
                                     llrq0,llrq0,llrq0,llrq0,0};
    double factorMessagesZ[14*3] = {};
    double m0 = 2*atanh(tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrq0/2));   //+1 stab to qubit error variable
    double m1 = -1*m0;                                                              //-1 stab to qubit error variable
    double m2 = 2*atanh(tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrp0/2));   //+1 stab to measurement error variable
    double m3 = -1*m2;                                                              //-1 stab to measurement error variable
    double m4 = 2*atanh(tanh(llrq0/2)*tanh(llrq0/2)*tanh(llrq0/2));                 //+1 metacheck to measurement error variable
    double m5 = -1*m4;                                                              //-1 metacheck to measurement error variable
    double factorMessagesZExpected[14*3] = {m1,m0,m1,
                                            m1,m0,m0,
                                            m1,m1,m1,
                                            m1,m0,m1,
                                            m0,m1,m1,
                                            m0,m0,m1,
                                            m1,m1,m1,
                                            m0,m1,m1,
                                            m3,m5,m5,
                                            m2,m5,m4,
                                            m3,m5,m4,
                                            m2,m5,m4,
                                            m3,m5,m4,
                                            m2,m5,m5};

    updateFactorMessagesTanhWrap(14, 9, variableMessagesX, factorMessagesZ, factorsZ, 
            eightq3Dcolour.factorToVariablesZ, eightq3Dcolour.factorDegreesZ, eightq3Dcolour.maxFactorDegreeZ, 
            eightq3Dcolour.factorToPosZ, eightq3Dcolour.maxVariableDegreeX);
    for (int i=0; i<14*3; ++i) EXPECT_NEAR(factorMessagesZ[i], factorMessagesZExpected[i], 1e-15);
}

//------------------------------------------------------------

TEST(updateFactorMessagesMinSumTest, CorrectOutput)
{
    int factorsZ[9] = {1,0,1,0,1,1,1,1,0};
    double llrp0 = log10(0.99/0.01);
    double llrq0 = log10(0.98/0.02);
    double variableMessagesX[9*5] = {llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrq0,llrq0,llrq0,llrq0,0,
                                     llrq0,llrq0,llrq0,llrq0,0,
                                     llrq0,llrq0,llrq0,llrq0,0};
    double factorMessagesZ[14*3] = {};
    //using alpha = 0.5, no idea if this is good in practise but fine for testing function
    double m0 = 0.5*llrq0;                                  //+1 stab to qubit error variable
    double m1 = -1*m0;                                      //-1 stab to qubit error variable
    double m2 = 0.5*llrq0;                                  //+1 stab to measurement error variable
    double m3 = -1*m2;                                      //-1 stab to measurement error variable
    double m4 = 0.5*llrq0;                                  //+1 metacheck to measurement error variable
    double m5 = -1*m4;                                      //-1 metacheck to measurement error variable
    double factorMessagesZExpected[14*3] = {m1,m0,m1,
                                            m1,m0,m0,
                                            m1,m1,m1,
                                            m1,m0,m1,
                                            m0,m1,m1,
                                            m0,m0,m1,
                                            m1,m1,m1,
                                            m0,m1,m1,
                                            m3,m5,m5,
                                            m2,m5,m4,
                                            m3,m5,m4,
                                            m2,m5,m4,
                                            m3,m5,m4,
                                            m2,m5,m5};

    updateFactorMessagesMinSum(0.5, 14, 9, variableMessagesX, factorMessagesZ, factorsZ,
            eightq3Dcolour.factorToVariablesZ, eightq3Dcolour.factorDegreesZ, eightq3Dcolour.maxFactorDegreeZ,
            eightq3Dcolour.factorToPosZ, eightq3Dcolour.maxVariableDegreeX);
    for (int i=0; i<14*3; ++i) EXPECT_NEAR(factorMessagesZ[i], factorMessagesZExpected[i], 1e-15);
}

//------------------------------------------------------------

TEST(updateVariableMessagesTest, CorrectOutput)
{
    int factorsZ[9] = {1,0,1,0,1,1,1,1,0};
    double llrp0 = log10(0.99/0.01);
    double llrq0 = log10(0.98/0.02);
    double m0 = 2*atanh(tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrq0/2));   //+1 stab to qubit error variable
    double m1 = -1*m0;                                                              //-1 stab to qubit error variable
    double m2 = 2*atanh(tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrp0/2));   //+1 stab to measurement error variable
    double m3 = -1*m2;                                                              //-1 stab to measurement error variable
    double m4 = 2*atanh(tanh(llrq0/2)*tanh(llrq0/2)*tanh(llrq0/2));                 //+1 metacheck to measurement error variable
    double m5 = -1*m4;                                                              //-1 metacheck to measurement error variable
    double factorMessagesZ[14*3] = {m1,m0,m1,
                                    m1,m0,m0,
                                    m1,m1,m1,
                                    m1,m0,m1,
                                    m0,m1,m1,
                                    m0,m0,m1,
                                    m1,m1,m1,
                                    m0,m1,m1,
                                    m3,m5,m5,
                                    m2,m5,m4,
                                    m3,m5,m4,
                                    m2,m5,m4,
                                    m3,m5,m4,
                                    m2,m5,m5};

    double variableMessagesX[9*5] = {llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrp0,llrp0,llrp0,llrp0,llrq0,
                                     llrq0,llrq0,llrq0,llrq0,0,
                                     llrq0,llrq0,llrq0,llrq0,0,
                                     llrq0,llrq0,llrq0,llrq0,0};

    double variableMessagesXExpected[9*5] = {llrp0+m0+m1,llrp0+m0+m0,llrp0+m1+m1,llrp0+m0+m1,llrq0+m5+m5,
                                             llrp0+m1+m1,llrp0+m1+m0,llrp0+m1+m1,llrp0+m0+m1,llrq0+m5+m4,
                                             llrp0+m1+m0,llrp0+m1+m1,llrp0+m0+m1,llrp0+m1+m1,llrq0+m5+m4,
                                             llrp0+m1+m0,llrp0+m1+m1,llrp0+m0+m1,llrp0+m1+m1,llrq0+m5+m4,
                                             llrp0+m0+m0,llrp0+m1+m0,llrp0+m1+m1,llrp0+m0+m1,llrq0+m5+m4,
                                             llrp0+m0+m1,llrp0+m0+m0,llrp0+m1+m1,llrp0+m0+m1,llrq0+m5+m5,
                                             llrq0+m3+m5,llrq0+m2+m4,llrq0+m3+m4,llrq0+m2+m5,0,
                                             llrq0+m3+m5,llrq0+m3+m4,llrq0+m2+m4,llrq0+m2+m5,0,
                                             llrq0+m2+m5,llrq0+m3+m5,llrq0+m2+m5,llrq0+m3+m5,0};
    
    updateVariableMessagesWrap(14, 9, 8, factorMessagesZ, variableMessagesX, eightq3Dcolour.variableToFactorsX,
            eightq3Dcolour.variableDegreesX, eightq3Dcolour.maxVariableDegreeX, eightq3Dcolour.variableToPosX, 
            eightq3Dcolour.maxFactorDegreeZ, llrp0, llrq0);
    for (int i=0; i<9*5; ++i) EXPECT_NEAR(variableMessagesX[i], variableMessagesXExpected[i], 1e-15);
}

//------------------------------------------------------------

TEST(calcMarginalsWrapTest, CorrectOutput)
{
    double llrp0 = log10(0.99/0.01);
    double llrq0 = log10(0.98/0.02);
    double m0 = 2*atanh(tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrq0/2));   //+1 stab to qubit error variable
    double m1 = -1*m0;                                                              //-1 stab to qubit error variable
    double m2 = 2*atanh(tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrp0/2));   //+1 stab to measurement error variable
    double m3 = -1*m2;                                                              //-1 stab to measurement error variable
    double m4 = 2*atanh(tanh(llrq0/2)*tanh(llrq0/2)*tanh(llrq0/2));                 //+1 metacheck to measurement error variable
    double m5 = -1*m4;                                                              //-1 metacheck to measurement error variable
    double factorMessagesZ[14*3] = {m1,m0,m1,
                                    m1,m0,m0,
                                    m1,m1,m1,
                                    m1,m0,m1,
                                    m0,m1,m1,
                                    m0,m0,m1,
                                    m1,m1,m1,
                                    m0,m1,m1,
                                    m3,m5,m5,
                                    m2,m5,m4,
                                    m3,m5,m4,
                                    m2,m5,m4,
                                    m3,m5,m4,
                                    m2,m5,m5};
    double marginalsX[14] = {};
    double marginalsXExpected[14] = {llrp0+m1+m0+m1,
                                     llrp0+m1+m0+m0,
                                     llrp0+m1+m1+m1,
                                     llrp0+m1+m0+m1,
                                     llrp0+m0+m1+m1,
                                     llrp0+m0+m0+m1,
                                     llrp0+m1+m1+m1,
                                     llrp0+m0+m1+m1,
                                     llrp0+m3+m5+m5,
                                     llrp0+m2+m5+m4,
                                     llrp0+m3+m5+m4,
                                     llrp0+m2+m5+m4,
                                     llrp0+m3+m5+m4,
                                     llrp0+m2+m5+m5};

    calcMarginalsWrap(14, 8, marginalsX, factorMessagesZ, eightq3Dcolour.variableDegreesX, 
                        eightq3Dcolour.maxVariableDegreeX, llrp0, llrq0);
    for (int i=0; i<14; ++i) EXPECT_NEAR(marginalsX[i], marginalsXExpected[i], 1e-15);
}

//------------------------------------------------------------

TEST(bpCorrectionTest, CorrectOutput1)
{
    int variablesX[14] = {0,0,1,0,0,0,0,0,0,0,0,0,0,1};
    int variablesXExpected[14] = {0,0,1,0,0,0,0,0,0,0,0,0,0,1};
    int factorsZ[9] = {1,0,1,0,1,1,1,1,0};
    int factorsZExpected[9] = {0,0,1,0,1,1,1,1,0};
    double llrp0 = log10(0.99/0.01);
    double llrq0 = log10(0.98/0.02);
    double m0 = 2*atanh(tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrq0/2));   //+1 stab to qubit error variable
    double m1 = -1*m0;                                                              //-1 stab to qubit error variable
    double m2 = 2*atanh(tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrp0/2)*tanh(llrp0/2));   //+1 stab to measurement error variable
    double m3 = -1*m2;                                                              //-1 stab to measurement error variable
    double m4 = 2*atanh(tanh(llrq0/2)*tanh(llrq0/2)*tanh(llrq0/2));                 //+1 metacheck to measurement error variable
    double m5 = -1*m4;                                                              //-1 metacheck to measurement error variable
    double marginalsX[14] = {llrp0+m1+m0+m1,
                             llrp0+m1+m0+m0,
                             llrp0+m1+m1+m1,
                             llrp0+m1+m0+m1,
                             llrp0+m0+m1+m1,
                             llrp0+m0+m0+m1,
                             llrp0+m1+m1+m1,
                             llrp0+m0+m1+m1,
                             llrp0+m3+m5+m5,
                             llrp0+m2+m5+m4,
                             llrp0+m3+m5+m4,
                             llrp0+m2+m5+m4,
                             llrp0+m3+m5+m4,
                             llrp0+m2+m5+m5};

    bpCorrectionWrap(14, 9, 8, 6, marginalsX, variablesX, factorsZ, 
            eightq3Dcolour.variableToFactorsX, eightq3Dcolour.variableDegreesX, eightq3Dcolour.maxVariableDegreeX);
    for (int i=0; i<14; ++i) EXPECT_EQ(variablesX[i], variablesXExpected[i]);
    for (int i=0; i<9; ++i) EXPECT_EQ(factorsZ[i], factorsZExpected[i]);
}
TEST(bpCorrectionTest, CorrectOutput2)
{
    int variablesX[14] = {1,0,0,0,0,0,0,0,0,0,0,0,0,0};
    int variablesXExpected[14] = {};
    int factorsZ[9] = {1,1,1,0,0,0,0,0,0};
    int factorsZExpected[9] = {};
    double marginalsX[14] = {-2,1,1,1,1,1,1,1,1,1,1,1,1};
    bpCorrectionWrap(14, 9, 8, 6, marginalsX, variablesX, factorsZ, 
            eightq3Dcolour.variableToFactorsX, eightq3Dcolour.variableDegreesX, eightq3Dcolour.maxVariableDegreeX);
    for (int i=0; i<14; ++i) EXPECT_EQ(variablesX[i], variablesXExpected[i]);
    for (int i=0; i<9; ++i) EXPECT_EQ(factorsZ[i], factorsZExpected[i]);
}
