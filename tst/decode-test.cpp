#include "gtest/gtest.h"
#include "decode_wrappers.cuh"
#include "prepCode.h"
#include<random>

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

TEST(arrayErrorsTest, NonTrivialOutput)
{
    //As always testing random functions is a pain 
    //so just check it does something
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    std::random_device rd{};
    arrayErrorsWrap(3*6*6*6, rd(), qubits, 0.5);
    arrayErrorsWrap(3*6*6*6, rd(), syndrome, 0.5);
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
    arrayErrorsWrap(3*6*6*6, rd(), qubits, 1);
    arrayErrorsWrap(3*6*6*6, rd(), syndrome, 1);
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

TEST(flipTest, surface)
{
    int qubitsX[4] = {1,0,0,0};
    int qubitsZ[4] = {1,0,0,0};
    int syndromeZ[2] = {1,0};
    int syndromeX[1] = {1};
    int qubitsXExpected[4] = {0,0,1,0};
    int qubitsZExpected[4] = {0,1,1,1};
    int syndromeZExpected[2] = {1,0};
    int syndromeXExpected[1] = {1};
    flipWrap(4, 2, qubitsX, syndromeZ, fourqsurface.bitToZChecks, fourqsurface.maxBitDegreeZ);
    flipWrap(4, 1, qubitsZ, syndromeX, fourqsurface.bitToXChecks, fourqsurface.maxBitDegreeX);
    for (int i=0; i<4; ++i)
    {
        EXPECT_EQ(qubitsX[i], qubitsXExpected[i]);
        EXPECT_EQ(qubitsZ[i], qubitsZExpected[i]);
    }
    EXPECT_EQ(syndromeZ[0], syndromeZExpected[0]);
    EXPECT_EQ(syndromeZ[1], syndromeZExpected[1]);
    EXPECT_EQ(syndromeX[0], syndromeXExpected[0]);
}
TEST(flipTest, colour)
{
    int qubitsX[7] = {0,0,0,0,1,0,0};
    int qubitsZ[7] = {0,1,0,0,0,0,0};
    int syndromeZ[3] = {1,1,1};
    int syndromeX[3] = {1,1,0};
    int qubitsXExpected[7] = {1,1,1,1,0,1,1};
    int qubitsZExpected[7] = {1,0,1,0,1,0,0};
    int syndromeZExpected[3] = {1,1,1};
    int syndromeXExpected[3] = {0,0,1};
    flipWrap(7, 3, qubitsX, syndromeZ, sevenqcolour.bitToZChecks, sevenqcolour.maxBitDegreeZ);
    flipWrap(7, 3, qubitsZ, syndromeX, sevenqcolour.bitToXChecks, sevenqcolour.maxBitDegreeX);
    for (int i=0; i<7; ++i)
    {
        EXPECT_EQ(qubitsX[i], qubitsXExpected[i]);
        EXPECT_EQ(qubitsZ[i], qubitsZExpected[i]);
    }
    for (int i=0; i<3; ++i)
    {
        EXPECT_EQ(syndromeZ[i], syndromeZExpected[i]);
        EXPECT_EQ(syndromeX[i], syndromeXExpected[i]);
    }
}

//------------------------------------------------------------

TEST(pflipTest, surface)
{
    //This one identical to flip because each qubit is part of only one stab of each type
    int qubitsX[4] = {1,0,0,0};
    int qubitsZ[4] = {1,0,0,0};
    int syndromeZ[2] = {1,0};
    int syndromeX[1] = {1};
    int qubitsXExpected[4] = {0,0,1,0};
    int qubitsZExpected[4] = {0,1,1,1};
    int syndromeZExpected[2] = {1,0};
    int syndromeXExpected[1] = {1};
    std::random_device rd{};
    pflipWrap(4, 2, rd(), qubitsX, syndromeZ, fourqsurface.bitToZChecks, fourqsurface.maxBitDegreeZ);
    pflipWrap(4, 1, rd(), qubitsZ, syndromeX, fourqsurface.bitToXChecks, fourqsurface.maxBitDegreeX);
    for (int i=0; i<4; ++i)
    {
        EXPECT_EQ(qubitsX[i], qubitsXExpected[i]);
        EXPECT_EQ(qubitsZ[i], qubitsZExpected[i]);
    }
    EXPECT_EQ(syndromeZ[0], syndromeZExpected[0]);
    EXPECT_EQ(syndromeZ[1], syndromeZExpected[1]);
    EXPECT_EQ(syndromeX[0], syndromeXExpected[0]);
}
TEST(pflipTest, colour)
{
    int qubitsX[7] = {0,0,0,0,1,0,0};
    int qubitsZ[7] = {0,1,0,0,0,0,0};
    int syndromeZ[3] = {1,1,1};
    int syndromeX[3] = {1,1,0};
    int qubitsXExpected[7] = {1,1,1,1,0,1,1};
    int qubitsZExpected1[7] = {1,0,1,0,1,0,0};
    int qubitsZExpected2[7] = {1,0,1,1,1,0,0};
    int qubitsZExpected3[7] = {1,0,1,0,1,1,0};
    int qubitsZExpected4[7] = {1,0,1,1,1,1,0};
    int syndromeZExpected[3] = {1,1,1};
    int syndromeXExpected1[3] = {0,0,1};
    int syndromeXExpected2[3] = {1,0,0};
    int syndromeXExpected3[3] = {0,1,0};
    int syndromeXExpected4[3] = {1,1,1};
    std::random_device rd{};
    pflipWrap(7, 3, rd(), qubitsX, syndromeZ, sevenqcolour.bitToZChecks, sevenqcolour.maxBitDegreeZ);
    pflipWrap(7, 3, rd(), qubitsZ, syndromeX, sevenqcolour.bitToXChecks, sevenqcolour.maxBitDegreeX);
    int matches[4] = {1,1,1,1};
    for (int i=0; i<7; ++i)
    {
        EXPECT_EQ(qubitsX[i], qubitsXExpected[i]);
        if (qubitsZ[i] != qubitsZExpected1[i]) matches[0] = 0;
        if (qubitsZ[i] != qubitsZExpected2[i]) matches[1] = 0;
        if (qubitsZ[i] != qubitsZExpected3[i]) matches[2] = 0;
        if (qubitsZ[i] != qubitsZExpected4[i]) matches[3] = 0;
    }
    int match = matches[0] + matches[1] + matches[2] + matches[3];
    EXPECT_EQ(match, 1);
    for (int i=0; i<4; ++i) matches[i] = 1;
    for (int i=0; i<3; ++i)
    {
        EXPECT_EQ(syndromeZ[i], syndromeZExpected[i]);
        if (syndromeX[i] != syndromeXExpected1[i]) matches[0] = 0;
        if (syndromeX[i] != syndromeXExpected2[i]) matches[1] = 0;
        if (syndromeX[i] != syndromeXExpected3[i]) matches[2] = 0;
        if (syndromeX[i] != syndromeXExpected4[i]) matches[3] = 0;
    }
    match = matches[0] + matches[1] + matches[2] + matches[3];
    EXPECT_EQ(match,1);
}

//------------------------------------------------------------

//qubits should never change from this function
TEST(calculateSyndromeTest, surface)
{
    int qubitsX[4] = {1,0,0,0};
    int qubitsZ[4] = {1,0,0,0};
    int syndromeZ[2] = {};
    int syndromeX[1] = {};
    int qubitsXExpected[4] = {1,0,0,0};
    int qubitsZExpected[4] = {1,0,0,0};
    int syndromeZExpected[2] = {1,0};
    int syndromeXExpected[1] = {1};
    calculateSyndromeWrap(4, 2, qubitsX, syndromeZ, fourqsurface.zCheckToBits, fourqsurface.maxCheckDegreeZ);
    calculateSyndromeWrap(4, 1, qubitsZ, syndromeX, fourqsurface.xCheckToBits, fourqsurface.maxCheckDegreeX);
    for (int i=0; i<4; ++i)
    {
        EXPECT_EQ(qubitsX[i], qubitsXExpected[i]);
        EXPECT_EQ(qubitsZ[i], qubitsZExpected[i]);
    }
    EXPECT_EQ(syndromeZ[0], syndromeZExpected[0]);
    EXPECT_EQ(syndromeZ[1], syndromeZExpected[1]);
    EXPECT_EQ(syndromeX[0], syndromeXExpected[0]);
}
TEST(calculateSyndromeTest, colour)
{
    int qubitsX[7] = {0,0,0,0,1,0,0};
    int qubitsZ[7] = {0,1,0,0,0,0,0};
    int syndromeZ[3] = {};
    int syndromeX[3] = {};
    int qubitsXExpected[7] = {0,0,0,0,1,0,0};
    int qubitsZExpected[7] = {0,1,0,0,0,0,0};
    int syndromeZExpected[3] = {1,1,1};
    int syndromeXExpected[3] = {1,1,0};
    calculateSyndromeWrap(7, 3, qubitsX, syndromeZ, sevenqcolour.zCheckToBits, sevenqcolour.maxCheckDegreeZ);
    calculateSyndromeWrap(7, 3, qubitsZ, syndromeX, sevenqcolour.xCheckToBits, sevenqcolour.maxCheckDegreeX);
    for (int i=0; i<7; ++i)
    {
        EXPECT_EQ(qubitsX[i], qubitsXExpected[i]);
        EXPECT_EQ(qubitsZ[i], qubitsZExpected[i]);
    }
    for (int i=0; i<3; ++i)
    {
        EXPECT_EQ(syndromeZ[i], syndromeZExpected[i]);
        EXPECT_EQ(syndromeX[i], syndromeXExpected[i]);
    }
}
