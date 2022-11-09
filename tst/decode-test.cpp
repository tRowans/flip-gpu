#include "gtest/gtest.h"
#include "decode_wrappers.cuh"
#include "prepCode.h"
#include<random>
#include<typeinfo>

//------------------------------------------------------------

TEST(wipeArrayTest, CorrectOutput)
{
    int arr[3*6*6*6];
    for (int i=0; i<3*6*6*6; ++i)
    {
        arr[i] = 1;
    }
    wipeArrayWrap(3*6*6*6, arr);
    for(int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(arr[i], 0);
    }
}

//------------------------------------------------------------

TEST(applyErrorsTest, NonTrivialOutput)
{
    //As always testing random functions is a pain 
    //so just check it does something
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    std::random_device rd{};
    applyErrorsWrap(3*6*6*6, rd(), testCodeC.qubitInclusionLookup, qubits, 0.5);
    applyErrorsWrap(3*6*6*6, rd(), testCodeC.stabInclusionLookup, syndrome, 0.5);
    int totalQ = 0;
    int totalS = 0;
    for (int i=0; i<3*6*6*6; ++i)
    {
        totalQ += qubits[i];
        totalS += syndrome[i];
    }
    EXPECT_NE(totalQ, 0);
    EXPECT_NE(totalS, 0);
}
TEST(applyErrorsTest, ProbabilityOne)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    std::random_device rd{};
    applyErrorsWrap(3*6*6*6, rd(), testCodeC.qubitInclusionLookup, qubits, 1);
    applyErrorsWrap(3*6*6*6, rd(), testCodeC.stabInclusionLookup, syndrome, 1);
    int totalQ = 0;
    int totalS = 0;
    for (int i=0; i<3*6*6*6; ++i)
    {
        totalQ += qubits[i];
        totalS += syndrome[i];
    }
    EXPECT_EQ(totalQ, 3*6*6*6);
    EXPECT_EQ(totalS, 3*6*6*6);
}

//------------------------------------------------------------

TEST(flipTest, SingleError)
{
    //Should correct this error
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};
    for (int i=0; i<3*6*6*6; ++i)
    {
        //This is just a trick to make flip run normally, don't want to test this right now
        qubitMarginals[i] = 1.0;
    }
    qubits[0] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[4] = 1;
    syndrome[18] = 1;
    flipWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges, testCodeC.edgeToFaces, qubitMessages, qubitMarginals);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
    }
}
TEST(flipTest, TwoErrors)
{
    //Should correct this error
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};
    for (int i=0; i<3*6*6*6; ++i)
    {
        qubitMarginals[i] = 1.0;
    }
    qubits[0] = 1;
    qubits[3] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[3] = 1;
    syndrome[7] = 1;
    syndrome[18] = 1;
    syndrome[21] = 1;
    flipWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges, testCodeC.edgeToFaces, qubitMessages, qubitMarginals);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
    }
}
TEST(flipTest, ThreeErrors)
{
    //Should partially correct this error (leaves one qubit)
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};
    for (int i=0; i<3*6*6*6; ++i)
    {
        qubitMarginals[i] = 1.0;
    }
    qubits[0] = 1;
    qubits[3] = 1;
    qubits[21] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[3] = 1;
    syndrome[7] = 1;
    syndrome[18] = 1;
    syndrome[22] = 1;
    syndrome[25] = 1;
    syndrome[39] = 1;
    qubitsExpected[3] = 1;
    syndromeExpected[3] = 1;
    syndromeExpected[4] = 1;
    syndromeExpected[7] = 1;
    syndromeExpected[21] = 1;
    flipWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges, testCodeC.edgeToFaces, qubitMessages, qubitMarginals);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
    }
}
TEST(flipTest, FourErrors)
{
    //Should leave this error unchanged
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};
    for (int i=0; i<3*6*6*6; ++i)
    {
        qubitMarginals[i] = 1.0;
    }
    qubits[0] = 1;
    qubits[3] = 1;
    qubits[18] = 1;
    qubits[21] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[3] = 1;
    syndrome[7] = 1;
    syndrome[19] = 1;
    syndrome[25] = 1;
    syndrome[36] = 1;
    syndrome[39] = 1;
    qubitsExpected[0] = 1;
    qubitsExpected[3] = 1;
    qubitsExpected[18] = 1;
    qubitsExpected[21] = 1;
    syndromeExpected[0] = 1;
    syndromeExpected[1] = 1;
    syndromeExpected[3] = 1;
    syndromeExpected[7] = 1;
    syndromeExpected[19] = 1;
    syndromeExpected[25] = 1;
    syndromeExpected[36] = 1;
    syndromeExpected[39] = 1;
    flipWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges, testCodeC.edgeToFaces, qubitMessages, qubitMarginals);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
    }
}
TEST(flipTest, CheckBpFunctionality)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};
    qubits[0] = 1;
    qubits[6] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[4] = 1;
    syndrome[18] = 1;
    syndrome[6] = 1;
    syndrome[7] = 1;
    syndrome[10] = 1;
    syndrome[24] = 1;
    qubitsExpected[6] = 1;
    syndromeExpected[6] = 1;
    syndromeExpected[7] = 1;
    syndromeExpected[10] = 1;
    syndromeExpected[24] = 1;
    qubitMarginals[0] = 1.0;
    //Should swap these
    qubitMessages[8*0+2*0] = 0.0;        //Messages from q0 to s0
    qubitMessages[8*0+2*0+1] = 1.0;
    qubitMessages[8*1+2*1] = 0.0;        //Messages from q0 to s1
    qubitMessages[8*1+2*1+1] = 1.0;
    qubitMessages[8*4+2*3] = 0.0;        //Messages from q0 to s4
    qubitMessages[8*4+2*3+1] = 1.0;
    qubitMessages[8*18+2*2] = 0.0;       //Messages from q0 to s18
    qubitMessages[8*18+2*2+1] = 1.0;
    //Should leave these the same
    qubitMessages[8*6+2*0] = 0.0;        //Messages from q6 to s6
    qubitMessages[8*6+2*0+1] = 1.0;
    qubitMessages[8*7+2*1] = 0.0;        //Messages from q6 to s7
    qubitMessages[8*7+2*1+1] = 1.0;
    qubitMessages[8*10+2*3] = 0.0;       //Messages from q6 to s10
    qubitMessages[8*10+2*3+1] = 1.0;
    qubitMessages[8*24+2*2] = 0.0;       //Messages from q6 to s24
    qubitMessages[8*24+2*2+1] = 1.0;
    flipWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges, testCodeC.edgeToFaces, qubitMessages, qubitMarginals);
    for (int i=0; i<8*3*6*6*6; ++i)
    {
        if (i < 3*6*6*6)
        {
            EXPECT_EQ(qubits[i], qubitsExpected[i]);
            EXPECT_EQ(syndrome[i], syndromeExpected[i]);
            if (i == 0) EXPECT_FLOAT_EQ(qubitMarginals[i], 1.0);
            else EXPECT_FLOAT_EQ(qubitMarginals[i], 0.0);
        }
        if (i == 0 || i == 10 || i == 38 || i == 148 ||
            i == 49 || i == 59 || i == 87 || i == 197)
        {
            EXPECT_FLOAT_EQ(qubitMessages[i], 1.0);
        }
        else EXPECT_FLOAT_EQ(qubitMessages[i], 0.0);
    }
}

//------------------------------------------------------------

TEST(pflipTest, OneError)
{
    //Should be the same as normal flip
    std::random_device rd{};
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};
    for (int i=0; i<3*6*6*6; ++i)
    {
        qubitMarginals[i] = 1.0;
    }
    qubits[0] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[4] = 1;
    syndrome[18] = 1;
    pflipWrap(3*6*6*6, rd(), testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges, testCodeC.edgeToFaces, qubitMessages, qubitMarginals);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
    }
}
TEST(pflipTest, TwoErrors)
{
    //Should also be the same as normal flip
    std::random_device rd{};
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};
    for (int i=0; i<3*6*6*6; ++i)
    {
        qubitMarginals[i] = 1.0;
    }
    qubits[0] = 1;
    qubits[3] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[3] = 1;
    syndrome[7] = 1;
    syndrome[18] = 1;
    syndrome[21] = 1;
    pflipWrap(3*6*6*6, rd(), testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges, testCodeC.edgeToFaces, qubitMessages, qubitMarginals);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
    }
}
TEST(pflipTest, ThreeErrors)
{
    //Should correct two errors and maybe affect two other qubits
    std::random_device rd{};
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};
    for (int i=0; i<3*6*6*6; ++i)
    {
        qubitMarginals[i] = 1.0;
    }
    qubits[0] = 1;
    qubits[3] = 1;
    qubits[21] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[3] = 1;
    syndrome[7] = 1;
    syndrome[18] = 1;
    syndrome[22] = 1;
    syndrome[25] = 1;
    syndrome[39] = 1;
    int qOptions[4][3*6*6*6] = {};
    qOptions[1][3] = 1;
    qOptions[2][18] = 1;
    qOptions[3][3] = 1;
    qOptions[3][18] = 1;
    int sOptions[4][3*6*6*6] = {};
    sOptions[1][3] = 1;
    sOptions[1][4] = 1;
    sOptions[1][7] = 1;
    sOptions[1][21] = 1;
    sOptions[2][18] = 1;
    sOptions[2][19] = 1;
    sOptions[2][22] = 1;
    sOptions[2][36] = 1;
    sOptions[3][3] = 1;
    sOptions[3][4] = 1;
    sOptions[3][7] = 1;
    sOptions[3][21] = 1;
    sOptions[3][18] = 1;
    sOptions[3][19] = 1;
    sOptions[3][22] = 1;
    sOptions[3][36] = 1;
    pflipWrap(3*6*6*6, rd(), testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges, testCodeC.edgeToFaces, qubitMessages, qubitMarginals);
    int qChecks[4] = {};
    int sChecks[4] = {};
    for (int i=0; i<4; ++i)
    {
       for (int j=0; j<3*6*6*6; ++j)
       {
           if (qubits[j] != qOptions[i][j]) qChecks[i] = 1;
           if (syndrome[j] != sOptions[i][j]) sChecks[i] = 1;
       }
    }
    EXPECT_EQ(qChecks[0]+qChecks[1]+qChecks[2]+qChecks[3],3);
    EXPECT_EQ(sChecks[0]+sChecks[1]+sChecks[2]+sChecks[3],3);
}
TEST(pflipTest, CheckBpFunctionality)
{
    std::random_device rd{};
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};
    qubits[0] = 1;
    qubits[6] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[4] = 1;
    syndrome[18] = 1;
    syndrome[6] = 1;
    syndrome[7] = 1;
    syndrome[10] = 1;
    syndrome[24] = 1;
    qubitsExpected[6] = 1;
    syndromeExpected[6] = 1;
    syndromeExpected[7] = 1;
    syndromeExpected[10] = 1;
    syndromeExpected[24] = 1;
    qubitMarginals[0] = 1.0;
    //Should swap these
    qubitMessages[8*0+2*0] = 0.0;        //Messages from q0 to s0
    qubitMessages[8*0+2*0+1] = 1.0;
    qubitMessages[8*1+2*1] = 0.0;        //Messages from q0 to s1
    qubitMessages[8*1+2*1+1] = 1.0;
    qubitMessages[8*4+2*3] = 0.0;        //Messages from q0 to s4
    qubitMessages[8*4+2*3+1] = 1.0;
    qubitMessages[8*18+2*2] = 0.0;       //Messages from q0 to s18
    qubitMessages[8*18+2*2+1] = 1.0;
    //Should leave these the same
    qubitMessages[8*6+2*0] = 0.0;        //Messages from q6 to s6
    qubitMessages[8*6+2*0+1] = 1.0;
    qubitMessages[8*7+2*1] = 0.0;        //Messages from q6 to s7
    qubitMessages[8*7+2*1+1] = 1.0;
    qubitMessages[8*10+2*3] = 0.0;       //Messages from q6 to s10
    qubitMessages[8*10+2*3+1] = 1.0;
    qubitMessages[8*24+2*2] = 0.0;       //Messages from q6 to s24
    qubitMessages[8*24+2*2+1] = 1.0;
    pflipWrap(3*6*6*6, rd(), testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges, testCodeC.edgeToFaces, qubitMessages, qubitMarginals);
    for (int i=0; i<8*3*6*6*6; ++i)
    {
        if (i < 3*6*6*6)
        {
            EXPECT_EQ(qubits[i], qubitsExpected[i]);
            EXPECT_EQ(syndrome[i], syndromeExpected[i]);
            if (i == 0) EXPECT_FLOAT_EQ(qubitMarginals[i], 1.0);
            else EXPECT_FLOAT_EQ(qubitMarginals[i], 0.0);
        }
        if (i == 0 || i == 10 || i == 38 || i == 148 ||
            i == 49 || i == 59 || i == 87 || i == 197)
        {
            EXPECT_FLOAT_EQ(qubitMessages[i], 1.0);
        }
        else EXPECT_FLOAT_EQ(qubitMessages[i], 0.0);
    }
}

//------------------------------------------------------------

//qubits should never change from this function
TEST(calculateSyndromeTest, OneError)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    qubits[0] = 1;
    qubitsExpected[0] = 1;
    syndromeExpected[0] = 1;
    syndromeExpected[1] = 1;
    syndromeExpected[4] = 1;
    syndromeExpected[18] = 1;
    calculateSyndromeWrap(3*6*6*6, testCodeC.stabInclusionLookup, qubits, syndrome, testCodeC.edgeToFaces);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
    }
}
TEST(calculateSyndromeTest, TwoErrors)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    qubits[0] = 1;
    qubits[3] = 1;
    qubitsExpected[0] = 1;
    qubitsExpected[3] = 1;
    syndromeExpected[0] = 1;
    syndromeExpected[1] = 1;
    syndromeExpected[3] = 1;
    syndromeExpected[7] = 1;
    syndromeExpected[18] = 1;
    syndromeExpected[21] = 1;
    calculateSyndromeWrap(3*6*6*6, testCodeC.stabInclusionLookup, qubits, syndrome, testCodeC.edgeToFaces);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
    }
}
TEST(calculateSyndromeTest, ThreeErrors)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    qubits[0] = 1;
    qubits[3] = 1;
    qubits[21] = 1;
    qubitsExpected[0] = 1;
    qubitsExpected[3] = 1;
    qubitsExpected[21] = 1;
    syndromeExpected[0] = 1;
    syndromeExpected[1] = 1;
    syndromeExpected[3] = 1;
    syndromeExpected[7] = 1;
    syndromeExpected[18] = 1;
    syndromeExpected[22] = 1;
    syndromeExpected[25] = 1;
    syndromeExpected[39] = 1;
    calculateSyndromeWrap(3*6*6*6, testCodeC.stabInclusionLookup, qubits, syndrome, testCodeC.edgeToFaces);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
    }
}
TEST(calculateSyndromeTest, FourErrors)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    qubits[0] = 1;
    qubits[3] = 1;
    qubits[18] = 1;
    qubits[21] = 1;
    qubitsExpected[0] = 1;
    qubitsExpected[3] = 1;
    qubitsExpected[18] = 1;
    qubitsExpected[21] = 1;
    syndromeExpected[0] = 1;
    syndromeExpected[1] = 1;
    syndromeExpected[3] = 1;
    syndromeExpected[7] = 1;
    syndromeExpected[19] = 1;
    syndromeExpected[25] = 1;
    syndromeExpected[36] = 1;
    syndromeExpected[39] = 1;
    calculateSyndromeWrap(3*6*6*6, testCodeC.stabInclusionLookup, qubits, syndrome, testCodeC.edgeToFaces);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
    }
}
TEST(calculateSyndromeTest, ExistingSyndrome)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    qubits[0] = 1;
    qubitsExpected[0] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[4] = 1;
    syndrome[18] = 1;
    syndromeExpected[0] = 1;
    syndromeExpected[1] = 1;
    syndromeExpected[4] = 1;
    syndromeExpected[18] = 1;
    calculateSyndromeWrap(3*6*6*6, testCodeC.stabInclusionLookup, qubits, syndrome, testCodeC.edgeToFaces);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
    }
}
TEST(calculateSyndromeTest, ExistingSyndrome2)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    int qubitsExpected[3*6*6*6] = {};
    int syndromeExpected[3*6*6*6] = {};
    qubits[0] = 1;
    qubitsExpected[0] = 1;
    syndrome[50] = 1;
    syndrome[52] = 1;
    syndromeExpected[0] = 1;
    syndromeExpected[1] = 1;
    syndromeExpected[4] = 1;
    syndromeExpected[18] = 1;
    calculateSyndromeWrap(3*6*6*6, testCodeC.stabInclusionLookup, qubits, syndrome, testCodeC.edgeToFaces);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
    }
}

//------------------------------------------------------------

TEST(updateSyndromeMessagesTest, CorrectOutput)
{
    int syndrome[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float syndromeMessages[8*3*6*6*6] = {};
    float syndromeMessagesExpected[8*3*6*6*6] = {};
    syndrome[129] = 1;
    syndrome[131] = 1;
    syndrome[134] = 1;
    syndrome[237] = 1;
    float p = 0.01; //assuming 1% error chance
    float m0 = 0.970596; //this is 3p^2(1-p) + (1-p)^3. This message gets sent by a stabiliser with value 0 (1) to a qubit when we condition on qubit value 0 (1)
    float m1 = 0.029404; //this is 3p(1-p)^2 + p^3. This message gets sent by a stabiliser with value 0 (1) to a qubit when we condition on qubit value 1 (0)
    for (int i=0; i<3*6*6*6; ++i)
    {
        for (int j=0; j<4; ++j)
        {
            //assuming 1% error chance here
            qubitMessages[8*i+2*j] = 0.99;
            qubitMessages[8*i+2*j+1] = 0.01;

            if (i == 130)
            {
                syndromeMessagesExpected[8*i+2*j] = m1;
                syndromeMessagesExpected[8*i+2*j+1] = m0;
            }
            else if (i == 129 || i == 133 || i == 237)
            {
                if (j == 0)
                {
                    syndromeMessagesExpected[8*i+2*j] = m1;
                    syndromeMessagesExpected[8*i+2*j+1] = m0;
                }
                else
                {
                    syndromeMessagesExpected[8*i+2*j] = m0;
                    syndromeMessagesExpected[8*i+2*j+1] = m1;
                }
            }
            else if (i == 131 || i == 134 || i == 238)
            {
                if (j == 1)
                {
                    syndromeMessagesExpected[8*i+2*j] = m1;
                    syndromeMessagesExpected[8*i+2*j+1] = m0;
                }
                else
                {
                    syndromeMessagesExpected[8*i+2*j] = m0;
                    syndromeMessagesExpected[8*i+2*j+1] = m1;
                }
            }
            else if (i == 111 || i == 127 || i == 219)
            {
                if (j == 2)
                {
                    syndromeMessagesExpected[8*i+2*j] = m1;
                    syndromeMessagesExpected[8*i+2*j+1] = m0;
                }
                else
                {
                    syndromeMessagesExpected[8*i+2*j] = m0;
                    syndromeMessagesExpected[8*i+2*j+1] = m1;
                }
            }
            else if (i == 22 || i == 113 || i == 116)
            {
                if (j==3)
                {
                    syndromeMessagesExpected[8*i+2*j] = m1;
                    syndromeMessagesExpected[8*i+2*j+1] = m0;
                }
                else 
                {
                    syndromeMessagesExpected[8*i+2*j] = m0;
                    syndromeMessagesExpected[8*i+2*j+1] = m1;
                }
            }
            else 
            {
                syndromeMessagesExpected[8*i+2*j] = m0;
                syndromeMessagesExpected[8*i+2*j+1] = m1;
            }
        }
    }
    updateSyndromeMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, qubitMessages, syndrome, 
                                syndromeMessages, testCodeC.edgeToFaces, testCodeC.faceToEdges);
    for (int i=0; i<8*3*6*6*6; ++i)
    {
        EXPECT_FLOAT_EQ(syndromeMessages[i], syndromeMessagesExpected[i]);
    }
}

//------------------------------------------------------------

TEST(updateQubitMessagesTest, CorrectOutput)
{
    float qubitMessages[8*3*6*6*6] = {};
    float qubitMessagesExpected[8*3*6*6*6] = {};
    float syndromeMessages[8*3*6*6*6] = {};
    float m0 = 0.970596; 
    float m1 = 0.029404;

    for (int i=0; i<3*6*6*6; ++i)
    {
        for (int j=0; j<4; ++j)
        {
            //use messages from above
            if (i == 130)
            {
                syndromeMessages[8*i+2*j] = m1;
                syndromeMessages[8*i+2*j+1] = m0;
            }
            else if (i == 129 || i == 133 || i == 237)
            {
                if (j == 0)
                {
                    syndromeMessages[8*i+2*j] = m1;
                    syndromeMessages[8*i+2*j+1] = m0;
                }
                else
                {
                    syndromeMessages[8*i+2*j] = m0;
                    syndromeMessages[8*i+2*j+1] = m1;
                }
            }
            else if (i == 131 || i == 134 || i == 238)
            {
                if (j == 1)
                {
                    syndromeMessages[8*i+2*j] = m1;
                    syndromeMessages[8*i+2*j+1] = m0;
                }
                else
                {
                    syndromeMessages[8*i+2*j] = m0;
                    syndromeMessages[8*i+2*j+1] = m1;
                }
            }
            else if (i == 111 || i == 127 || i == 219)
            {
                if (j == 2)
                {
                    syndromeMessages[8*i+2*j] = m1;
                    syndromeMessages[8*i+2*j+1] = m0;
                }
                else
                {
                    syndromeMessages[8*i+2*j] = m0;
                    syndromeMessages[8*i+2*j+1] = m1;
                }
            }
            else if (i == 22 || i == 113 || i == 116)
            {
                if (j==3)
                {
                    syndromeMessages[8*i+2*j] = m1;
                    syndromeMessages[8*i+2*j+1] = m0;
                }
                else 
                {
                    syndromeMessages[8*i+2*j] = m0;
                    syndromeMessages[8*i+2*j+1] = m1;
                }
            }
            else 
            {
                syndromeMessages[8*i+2*j] = m0;
                syndromeMessages[8*i+2*j+1] = m1;
            }
        }
    }
    updateQubitMessagesWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubitMessages,
                             syndromeMessages, testCodeC.faceToEdges, testCodeC.edgeToFaces, 0.01);
    //just check some edges because checking the whole thing is too much of a headache
    float a0 = 0.99f*m0*m0*m0 + 0.01f*m1*m1*m1;
    float a1 = 0.99f*m1*m1*m1 + 0.01f*m0*m0*m0;
    float a2 = 0.99f*m0*m0*m1 + 0.01f*m0*m1*m1;
    //-1 stabiliser
    float entries129[8] = {(0.99f*m0*m0*m0)/a0, (0.01f*m1*m1*m1)/a0,
                         (0.99f*m1*m1*m1)/a1, (0.01f*m0*m0*m0)/a1,
                         (0.99f*m0*m0*m0)/a0, (0.01f*m1*m1*m1)/a0,
                         (0.99f*m0*m0*m0)/a0, (0.01f*m1*m1*m1)/a0};
    //shares a vertex with a -1 stabiliser
    float entries112[8] = {(0.99f*m0*m0*m1)/a2, (0.01f*m0*m1*m1)/a2,
                         (0.99f*m0*m0*m1)/a2, (0.01f*m0*m1*m1)/a2,
                         (0.99f*m0*m0*m0)/a0, (0.01f*m1*m1*m1)/a0,
                         (0.99f*m0*m0*m0)/a0, (0.01f*m1*m1*m1)/a0};
    //disconnected from all -1 stabilisers
    float entries128[8] = {(0.99f*m0*m0*m1)/a2, (0.01f*m0*m1*m1)/a2,
                         (0.99f*m0*m0*m0)/a0, (0.01f*m1*m1*m1)/a0,
                         (0.99f*m0*m0*m0)/a0, (0.01f*m1*m1*m1)/a0,
                         (0.99f*m0*m0*m0)/a0, (0.01f*m1*m1*m1)/a0};
    for (int i=0; i<8; ++i)
    {
        EXPECT_FLOAT_EQ(entries129[i], qubitMessages[8*129+i]);
        EXPECT_FLOAT_EQ(entries112[i], qubitMessages[8*112+i]);
        EXPECT_FLOAT_EQ(entries128[i], qubitMessages[8*128+i]);
    }
}

//------------------------------------------------------------

TEST(calcMarginalsTest, CorrectOutput)
{
    float qubitMarginals[3*6*6*6] = {};
    float qubitMarginalsExpected[3*6*6*6] = {};
    float syndromeMessages[8*3*6*6*6] = {};
    float m0 = 0.970596; 
    float m1 = 0.029404;

    for (int i=0; i<3*6*6*6; ++i)
    {
        for (int j=0; j<4; ++j)
        {
            //use messages from above
            if (i == 130)
            {
                syndromeMessages[8*i+2*j] = m1;
                syndromeMessages[8*i+2*j+1] = m0;
            }
            else if (i == 129 || i == 133 || i == 237)
            {
                if (j == 0)
                {
                    syndromeMessages[8*i+2*j] = m1;
                    syndromeMessages[8*i+2*j+1] = m0;
                }
                else
                {
                    syndromeMessages[8*i+2*j] = m0;
                    syndromeMessages[8*i+2*j+1] = m1;
                }
            }
            else if (i == 131 || i == 134 || i == 238)
            {
                if (j == 1)
                {
                    syndromeMessages[8*i+2*j] = m1;
                    syndromeMessages[8*i+2*j+1] = m0;
                }
                else
                {
                    syndromeMessages[8*i+2*j] = m0;
                    syndromeMessages[8*i+2*j+1] = m1;
                }
            }
            else if (i == 111 || i == 127 || i == 219)
            {
                if (j == 2)
                {
                    syndromeMessages[8*i+2*j] = m1;
                    syndromeMessages[8*i+2*j+1] = m0;
                }
                else
                {
                    syndromeMessages[8*i+2*j] = m0;
                    syndromeMessages[8*i+2*j+1] = m1;
                }
            }
            else if (i == 22 || i == 113 || i == 116)
            {
                if (j==3)
                {
                    syndromeMessages[8*i+2*j] = m1;
                    syndromeMessages[8*i+2*j+1] = m0;
                }
                else 
                {
                    syndromeMessages[8*i+2*j] = m0;
                    syndromeMessages[8*i+2*j+1] = m1;
                }
            }
            else 
            {
                syndromeMessages[8*i+2*j] = m0;
                syndromeMessages[8*i+2*j+1] = m1;
            }
        }
    }
    float a0 = 0.99f*m0*m0*m0*m0 + 0.01f*m1*m1*m1*m1;
    float a1 = 0.99f*m1*m1*m1*m1 + 0.01f*m0*m0*m0*m0;
    float a2 = 0.99f*m0*m0*m0*m1 + 0.01f*m0*m1*m1*m1;
    for (int i=0; i<3*6*6*6; ++i)
    {
        if (i == 130) qubitMarginalsExpected[i] = (0.01*m0*m0*m0*m0)/a1;
        else if (i == 22 || i == 111 || i == 129
              || i == 113 || i == 127 || i == 131
              || i == 116 || i == 133 || i == 134
              || i == 219 || i == 237 || i == 238) qubitMarginalsExpected[i] = (0.01f*m0*m1*m1*m1)/a2;
        else qubitMarginalsExpected[i] = (0.01f*m1*m1*m1*m1)/a0;
    }
    calcMarginalsWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubitMarginals, syndromeMessages, 0.01);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_FLOAT_EQ(qubitMarginals[i], qubitMarginalsExpected[i]);
    }
}

//------------------------------------------------------------

TEST(fullBPTest, OneError)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float syndromeMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};

    qubits[0] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[4] = 1;
    syndrome[18] = 1;
    
    for (int i=0; i<4*3*6*6*6; ++i)
    {
        qubitMessages[2*i] = 0.99;
        qubitMessages[2*i+1] = 0.01;
    }

    for (int i=0; i<30; ++i)
    {
        updateSyndromeMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, qubitMessages, syndrome, 
                                    syndromeMessages, testCodeC.edgeToFaces, testCodeC.faceToEdges);
        updateQubitMessagesWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubitMessages, syndromeMessages,
                                    testCodeC.faceToEdges, testCodeC.edgeToFaces, 0.01);
    }
    calcMarginalsWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubitMarginals, syndromeMessages, 0.01);
    bpCorrectionWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubits, qubitMarginals);

    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], 0);
    }
}
TEST(fullBPTest, TwoErrors)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float syndromeMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};

    qubits[0] = 1;
    qubits[3] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[3] = 1;
    syndrome[7] = 1;
    syndrome[18] = 1;
    syndrome[21] = 1;

    for (int i=0; i<4*3*6*6*6; ++i)
    {
        qubitMessages[2*i] = 0.99;
        qubitMessages[2*i+1] = 0.01;
    }

    for (int i=0; i<30; ++i)
    {
        updateSyndromeMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, qubitMessages, syndrome, 
                                    syndromeMessages, testCodeC.edgeToFaces, testCodeC.faceToEdges);
        updateQubitMessagesWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubitMessages, syndromeMessages,
                                    testCodeC.faceToEdges, testCodeC.edgeToFaces, 0.01);
    }
    calcMarginalsWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubitMarginals, syndromeMessages, 0.01);
    bpCorrectionWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubits, qubitMarginals);

    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], 0);
    }
}
TEST(fullBPTest, ThreeErrors)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float syndromeMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};

    qubits[0] = 1;
    qubits[3] = 1;
    qubits[21] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[3] = 1;
    syndrome[7] = 1;
    syndrome[18] = 1;
    syndrome[22] = 1;
    syndrome[25] = 1;
    syndrome[39] = 1;

    for (int i=0; i<4*3*6*6*6; ++i)
    {
        qubitMessages[2*i] = 0.99;
        qubitMessages[2*i+1] = 0.01;
    }

    for (int i=0; i<30; ++i)
    {
        updateSyndromeMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, qubitMessages, syndrome, 
                                    syndromeMessages, testCodeC.edgeToFaces, testCodeC.faceToEdges);
        updateQubitMessagesWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubitMessages, syndromeMessages,
                                    testCodeC.faceToEdges, testCodeC.edgeToFaces, 0.01);
    }
    calcMarginalsWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubitMarginals, syndromeMessages, 0.01);
    bpCorrectionWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubits, qubitMarginals);

    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], 0);
    }
}
TEST(fullBPTest, FourErrors)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    float qubitMessages[8*3*6*6*6] = {};
    float syndromeMessages[8*3*6*6*6] = {};
    float qubitMarginals[3*6*6*6] = {};

    qubits[0] = 1;
    qubits[3] = 1;
    qubits[18] = 1;
    qubits[21] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[3] = 1;
    syndrome[7] = 1;
    syndrome[19] = 1;
    syndrome[25] = 1;
    syndrome[36] = 1;
    syndrome[39] = 1;

    for (int i=0; i<4*3*6*6*6; ++i)
    {
        qubitMessages[2*i] = 0.99;
        qubitMessages[2*i+1] = 0.01;
    }

    for (int i=0; i<30; ++i)
    {
        updateSyndromeMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, qubitMessages, syndrome, 
                                    syndromeMessages, testCodeC.edgeToFaces, testCodeC.faceToEdges);
        updateQubitMessagesWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubitMessages, syndromeMessages,
                                    testCodeC.faceToEdges, testCodeC.edgeToFaces, 0.01);
    }
    calcMarginalsWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubitMarginals, syndromeMessages, 0.01);
    bpCorrectionWrap(3*6*6*6, testCodeC.qubitInclusionLookup, qubits, qubitMarginals);

    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], 0);
    }
}

//------------------------------------------------------------

TEST(measureLogicalsTest, NoErrors)
{
    int qubits[3*6*6*6] = {};
    int nOdd;
    measureLogicalsWrap(3*6*6*6, testCodeC.logicalInclusionLookup, qubits, nOdd, 6, 'c');
    EXPECT_EQ(nOdd, 0);
}
TEST(measureLogicalsTest, OneError)
{
    int qubits[3*6*6*6] = {};
    qubits[0] = 1;
    int nOdd;
    measureLogicalsWrap(3*6*6*6, testCodeC.logicalInclusionLookup, qubits, nOdd, 6, 'c');
    EXPECT_EQ(nOdd, 1);
}
TEST(measureLogicalsTest, TwoErrorsSameLogical)
{
    int qubits[3*6*6*6] = {};
    qubits[0] = 1;
    qubits[108] = 1;
    int nOdd;
    measureLogicalsWrap(3*6*6*6, testCodeC.logicalInclusionLookup, qubits, nOdd, 6, 'c');
    EXPECT_EQ(nOdd, 0);
}
TEST(measureLogicalsTest, TwoErrorsDifferentLogicals)
{
    int qubits[3*6*6*6] = {};
    qubits[0] = 1;
    qubits[3] = 1;
    int nOdd;
    measureLogicalsWrap(3*6*6*6, testCodeC.logicalInclusionLookup, qubits, nOdd, 6, 'c');
    EXPECT_EQ(nOdd, 2);
}
TEST(measureLogicalsTest, ThreeErrors)
{
    int qubits[3*6*6*6] = {};
    qubits[0] = 1;
    qubits[3] = 1;
    qubits[18] = 1;
    int nOdd;
    measureLogicalsWrap(3*6*6*6, testCodeC.logicalInclusionLookup, qubits, nOdd, 6, 'c');
    EXPECT_EQ(nOdd, 3);
}
TEST(measureLogicalsTest, FourErrors)
{
    int qubits[3*6*6*6] = {};
    qubits[0] = 1;
    qubits[3] = 1;
    qubits[18] = 1;
    qubits[21] = 1;
    int nOdd;
    measureLogicalsWrap(3*6*6*6, testCodeC.logicalInclusionLookup, qubits, nOdd, 6, 'c');
    EXPECT_EQ(nOdd, 4);
}
TEST(measureLogicalsTest, LogicalX)
{
    int qubits[3*6*6*6] = {};
    qubits[0] = 1;
    qubits[3] = 1;
    qubits[6] = 1;
    qubits[9] = 1;
    qubits[12] = 1;
    qubits[15] = 1;
    qubits[18] = 1;
    qubits[21] = 1;
    qubits[24] = 1;
    qubits[27] = 1;
    qubits[30] = 1;
    qubits[33] = 1;
    qubits[36] = 1;
    qubits[39] = 1;
    qubits[42] = 1;
    qubits[45] = 1;
    qubits[48] = 1;
    qubits[51] = 1;
    qubits[54] = 1;
    qubits[57] = 1;
    qubits[60] = 1;
    qubits[63] = 1;
    qubits[66] = 1;
    qubits[69] = 1;
    qubits[72] = 1;
    qubits[75] = 1;
    qubits[78] = 1;
    qubits[81] = 1;
    qubits[84] = 1;
    qubits[87] = 1;
    qubits[90] = 1;
    qubits[93] = 1;
    qubits[96] = 1;
    qubits[99] = 1;
    qubits[102] = 1;
    qubits[105] = 1;
    int nOdd;
    measureLogicalsWrap(3*6*6*6, testCodeC.logicalInclusionLookup, qubits, nOdd, 6, 'c');
    EXPECT_EQ(nOdd, 36);
}
TEST(measureLogicalsTest, qubitAtTop)
{
    int qubits[3*6*6*6] = {};
    qubits[180] = 1;
    int nOdd;
    measureLogicalsWrap(3*6*6*6, testCodeC.logicalInclusionLookup, qubits, nOdd, 6, 'c');
    EXPECT_EQ(nOdd, 1);
}
TEST(measureLogicalsTest, noSidewaysQubits)
{
    int qubits[3*6*6*6] = {};
    qubits[1] = 1;
    int nOdd;
    measureLogicalsWrap(3*6*6*6, testCodeC.logicalInclusionLookup, qubits, nOdd, 6, 'c');
    EXPECT_EQ(nOdd, 0);
}
TEST(measureLogicalsTest, openBoundaries)
{
    int qubits[3*6*6*6] = {};
    int nOdd;
    measureLogicalsWrap(3*6*6*6, testCodeO.logicalInclusionLookup, qubits, nOdd, 6, 'o');
    EXPECT_EQ(nOdd, 0);
    qubits[0] = 1;
    measureLogicalsWrap(3*6*6*6, testCodeO.logicalInclusionLookup, qubits, nOdd, 6, 'o');
    EXPECT_EQ(nOdd, 1);
}
