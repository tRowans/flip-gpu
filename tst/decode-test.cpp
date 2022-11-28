#include "gtest/gtest.h"
#include "decode_wrappers.cuh"
#include "prepCode.h"
#include<random>
#include<typeinfo>
#include<math.h>

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
    qubits[0] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[4] = 1;
    syndrome[18] = 1;
    flipWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges);
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
    qubits[0] = 1;
    qubits[3] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[3] = 1;
    syndrome[7] = 1;
    syndrome[18] = 1;
    syndrome[21] = 1;
    flipWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges);
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
                qubits, syndrome, testCodeC.faceToEdges);
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
                qubits, syndrome, testCodeC.faceToEdges);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], qubitsExpected[i]);
        EXPECT_EQ(syndrome[i], syndromeExpected[i]);
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
    qubits[0] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[4] = 1;
    syndrome[18] = 1;
    pflipWrap(3*6*6*6, rd(), testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges);
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
    qubits[0] = 1;
    qubits[3] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[3] = 1;
    syndrome[7] = 1;
    syndrome[18] = 1;
    syndrome[21] = 1;
    pflipWrap(3*6*6*6, rd(), testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                qubits, syndrome, testCodeC.faceToEdges);
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
                qubits, syndrome, testCodeC.faceToEdges);
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

TEST(initVariableMessagesTest, CorrectOutput)
{
   double variableMessages[5*3*6*6*6];
   double llr0 = log10(0.99/0.01);
   initVariableMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, variableMessages, llr0, llr0);
   for (int i=0; i<5*3*6*6*6; ++i)
   {
       EXPECT_DOUBLE_EQ(variableMessages[i],llr0);
   }
}

//------------------------------------------------------------

TEST(updateFactorMessagesTest, CorrectOutput)
{
    int syndrome[3*6*6*6] = {};
    double variableMessages[5*3*6*6*6] = {};
    double factorMessages[5*3*6*6*6] = {};
    double factorMessagesExpected[5*3*6*6*6] = {};
    syndrome[129] = 1;
    syndrome[131] = 1;
    syndrome[134] = 1;
    syndrome[237] = 1;
    double llr0 = log10(0.99/0.01);    //assuming 1% error chance
    double m0 = 2*atanh(tanh(llr0/2)*tanh(llr0/2)*tanh(llr0/2)*tanh(llr0/2));
    double m1 = -1*m0;

    for (int i=0; i<3*6*6*6; ++i)
    {
        for (int j=0; j<5; ++j)
        {
            variableMessages[5*i+j] = llr0;
            
            if (j < 4)
            {
                if (i == 130) factorMessagesExpected[4*i+j] = m1;
                else if (i == 129 || i == 133 || i == 237)
                {
                    if (j == 0) factorMessagesExpected[4*i+j] = m1;
                    else factorMessagesExpected[4*i+j] = m0;
                }
                else if (i == 131 || i == 134 || i == 238)
                {
                    if (j == 1) factorMessagesExpected[4*i+j] = m1;
                    else factorMessagesExpected[4*i+j] = m0;
                }
                else if (i == 111 || i == 127 || i == 219)
                {
                    if (j == 2) factorMessagesExpected[4*i+j] = m1;
                    else factorMessagesExpected[4*i+j] = m0;
                }
                else if (i == 22 || i == 113 || i == 116)
                {
                    if (j==3) factorMessagesExpected[4*i+j] = m1;
                    else factorMessagesExpected[4*i+j] = m0;
                }
                else factorMessagesExpected[4*i+j] = m0;
            }
            else 
            {
                if (i == 129 || i == 131 || i == 134 || i == 237)
                { 
                    factorMessagesExpected[4*3*6*6*6+i] = m1;
                }
                else 
                {
                    factorMessagesExpected[4*3*6*6*6+i] = m0;
                }
            }
        }
    }
    updateFactorMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, variableMessages, syndrome, 
                                factorMessages, testCodeC.edgeToFaces, testCodeC.faceToEdges);
    for (int i=0; i<5*3*6*6*6; ++i)
    {
        EXPECT_NEAR(factorMessages[i], factorMessagesExpected[i], 1e-15);
    }
}

//------------------------------------------------------------

TEST(updateVariableMessagesTest, CorrectOutput)
{
    double variableMessages[5*3*6*6*6] = {};
    double variableMessagesExpected[5*3*6*6*6] = {};
    double factorMessages[5*3*6*6*6] = {};
    double llr0 = log10(0.99/0.01);    //assuming 1% error chance
    double m0 = 2*atanh(tanh(llr0/2)*tanh(llr0/2)*tanh(llr0/2)*tanh(llr0/2));
    double m1 = -1*m0;

    for (int i=0; i<3*6*6*6; ++i)
    {
        for (int j=0; j<5; ++j)
        {
            variableMessages[5*i+j] = llr0;
            
            if (j < 4)
            {
                if (i == 130) factorMessages[4*i+j] = m1;
                else if (i == 129 || i == 133 || i == 237)
                {
                    if (j == 0) factorMessages[4*i+j] = m1;
                    else factorMessages[4*i+j] = m0;
                }
                else if (i == 131 || i == 134 || i == 238)
                {
                    if (j == 1) factorMessages[4*i+j] = m1;
                    else factorMessages[4*i+j] = m0;
                }
                else if (i == 111 || i == 127 || i == 219)
                {
                    if (j == 2) factorMessages[4*i+j] = m1;
                    else factorMessages[4*i+j] = m0;
                }
                else if (i == 22 || i == 113 || i == 116)
                {
                    if (j==3) factorMessages[4*i+j] = m1;
                    else factorMessages[4*i+j] = m0;
                }
                else factorMessages[4*i+j] = m0;
            }
            else 
            {
                if (i == 129 || i == 131 || i == 134 || i == 237)
                { 
                    factorMessages[4*3*6*6*6+i] = m1;
                }
                else 
                {
                    factorMessages[4*3*6*6*6+i] = m0;
                }
            }
        }
    }
    updateVariableMessagesWrap(3*6*6*6, testCodeC.qubitInclusionLookup, variableMessages,
                             factorMessages, testCodeC.faceToEdges, testCodeC.edgeToFaces, llr0);
    //just check some edges because checking the whole thing is too much of a headache
    //-1 stabiliser
    double entries129[5] = {llr0+m0+m0+m0, llr0+m1+m1+m1, llr0+m0+m0+m0, llr0+m0+m0+m0, llr0};
    //shares a vertex with a -1 stabiliser
    double entries112[5] = {llr0+m0+m0+m1, llr0+m0+m0+m1, llr0+m0+m0+m0, llr0+m0+m0+m0, llr0};
    //disconnected from all -1 stabilisers
    double entries128[5] = {llr0+m0+m0+m1, llr0+m0+m0+m0, llr0+m0+m0+m0, llr0+m0+m0+m0, llr0};
    for (int i=0; i<5; ++i)
    {
        EXPECT_NEAR(entries129[i], variableMessages[5*129+i], 1e-15);
        EXPECT_NEAR(entries112[i], variableMessages[5*112+i], 1e-15);
        EXPECT_NEAR(entries128[i], variableMessages[5*128+i], 1e-15);
    }
}

//------------------------------------------------------------

TEST(calcMarginalsTest, CorrectOutput)
{ 
    double qubitMarginals[3*6*6*6] = {};
    double qubitMarginalsExpected[3*6*6*6] = {};
    double stabMarginals[3*6*6*6] = {};
    double stabMarginalsExpected[3*6*6*6] = {};
    double factorMessages[5*3*6*6*6] = {};
    double llr0 = log10(0.99/0.01);    //assuming 1% error chance
    double m0 = 2*atanh(tanh(llr0/2)*tanh(llr0/2)*tanh(llr0/2)*tanh(llr0/2));
    double m1 = -1*m0;

    for (int i=0; i<3*6*6*6; ++i)
    {
        for (int j=0; j<5; ++j)
        {
            if (j < 4)
            {
                if (i == 130) factorMessages[4*i+j] = m1;
                else if (i == 129 || i == 133 || i == 237)
                {
                    if (j == 0) factorMessages[4*i+j] = m1;
                    else factorMessages[4*i+j] = m0;
                }
                else if (i == 131 || i == 134 || i == 238)
                {
                    if (j == 1) factorMessages[4*i+j] = m1;
                    else factorMessages[4*i+j] = m0;
                }
                else if (i == 111 || i == 127 || i == 219)
                {
                    if (j == 2) factorMessages[4*i+j] = m1;
                    else factorMessages[4*i+j] = m0;
                }
                else if (i == 22 || i == 113 || i == 116)
                {
                    if (j==3) factorMessages[4*i+j] = m1;
                    else factorMessages[4*i+j] = m0;
                }
                else factorMessages[4*i+j] = m0;
            }
            else 
            {
                if (i == 129 || i == 131 || i == 134 || i == 237)
                { 
                    factorMessages[4*3*6*6*6+i] = m1;
                }
                else 
                {
                    factorMessages[4*3*6*6*6+i] = m0;
                }
            }
        }
    }
    //set expected qubit marginals
    for (int i=0; i<3*6*6*6; ++i)
    {
        if (i == 130) qubitMarginalsExpected[i] = llr0+m1+m1+m1+m1;
        else if (i == 22 || i == 111 || i == 129
              || i == 113 || i == 127 || i == 131
              || i == 116 || i == 133 || i == 134
              || i == 219 || i == 237 || i == 238) qubitMarginalsExpected[i] = llr0+m0+m0+m0+m1;
        else qubitMarginalsExpected[i] = llr0+m0+m0+m0+m0;
    }
    //set expected stab marginals
    for (int i=0; i<3*6*6*6; ++i)
    {
        if (i == 129 || i == 131 || i == 134 || i == 237) stabMarginalsExpected[i] = llr0 + m1;
        else stabMarginalsExpected[i] = llr0 + m0;
    }
    calcMarginalsWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                                              qubitMarginals, stabMarginals, factorMessages, llr0, llr0);
    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_NEAR(qubitMarginals[i], qubitMarginalsExpected[i], 1e-15);
        EXPECT_NEAR(stabMarginals[i], stabMarginalsExpected[i], 1e-15);
    }
}
//------------------------------------------------------------

TEST(fullBPTest, OneError)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    double variableMessages[5*3*6*6*6] = {};
    double factorMessages[5*3*6*6*6] = {};
    double qubitMarginals[3*6*6*6] = {};
    double stabMarginals[3*6*6*6] = {};

    qubits[0] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[4] = 1;
    syndrome[18] = 1;
    
    double llr0 = log10(0.99/0.01);    //assuming 1% error chance
    for (int i=0; i<5*3*6*6*6; ++i) variableMessages[i] = llr0;

    for (int i=0; i<30; ++i)
    {
        updateFactorMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, variableMessages, 
                       syndrome, factorMessages, testCodeC.edgeToFaces, testCodeC.faceToEdges);
        updateVariableMessagesWrap(3*6*6*6, testCodeC.qubitInclusionLookup, variableMessages, 
                       factorMessages, testCodeC.faceToEdges, testCodeC.edgeToFaces, llr0);
    }
    calcMarginalsWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup,
                       qubitMarginals, stabMarginals, factorMessages, llr0, llr0);
    bpCorrectionWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                       qubits, qubitMarginals, syndrome, stabMarginals, testCodeC.faceToEdges);

    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], 0);
        EXPECT_EQ(syndrome[i], 0);
    }
}
TEST(fullBPTest, TwoErrors)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    double variableMessages[5*3*6*6*6] = {};
    double factorMessages[5*3*6*6*6] = {};
    double qubitMarginals[3*6*6*6] = {};
    double stabMarginals[3*6*6*6] = {};

    qubits[0] = 1;
    qubits[3] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[3] = 1;
    syndrome[7] = 1;
    syndrome[18] = 1;
    syndrome[21] = 1;

    double llr0 = log10(0.99/0.01);    //assuming 1% error chance
    for (int i=0; i<5*3*6*6*6; ++i) variableMessages[i] = llr0;

    for (int i=0; i<30; ++i)
    {
        updateFactorMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, variableMessages, 
                      syndrome, factorMessages, testCodeC.edgeToFaces, testCodeC.faceToEdges);
        updateVariableMessagesWrap(3*6*6*6, testCodeC.qubitInclusionLookup, variableMessages, 
                      factorMessages, testCodeC.faceToEdges, testCodeC.edgeToFaces, llr0);
    }
    calcMarginalsWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup,
                      qubitMarginals, stabMarginals, factorMessages, llr0, llr0);
    bpCorrectionWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                      qubits, qubitMarginals, syndrome, stabMarginals, testCodeC.faceToEdges);

    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], 0);
        EXPECT_EQ(syndrome[i], 0);
    }
}
TEST(fullBPTest, ThreeErrors)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    double variableMessages[5*3*6*6*6] = {};
    double factorMessages[5*3*6*6*6] = {};
    double qubitMarginals[3*6*6*6] = {};
    double stabMarginals[3*6*6*6] = {};

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

    double llr0 = log10(0.99/0.01);    //assuming 1% error chance
    for (int i=0; i<5*3*6*6*6; ++i) variableMessages[i] = llr0;

    for (int i=0; i<30; ++i)
    {
        updateFactorMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, variableMessages, 
                        syndrome, factorMessages, testCodeC.edgeToFaces, testCodeC.faceToEdges);
        updateVariableMessagesWrap(3*6*6*6, testCodeC.qubitInclusionLookup, variableMessages, 
                        factorMessages, testCodeC.faceToEdges, testCodeC.edgeToFaces, llr0);
    }
    calcMarginalsWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                        qubitMarginals, stabMarginals, factorMessages, llr0, llr0);
    bpCorrectionWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                      qubits, qubitMarginals, syndrome, stabMarginals, testCodeC.faceToEdges);

    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], 0);
        EXPECT_EQ(syndrome[i], 0);
    }
}
TEST(fullBPTest, FourErrors)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    double variableMessages[5*3*6*6*6] = {};
    double factorMessages[5*3*6*6*6] = {};
    double qubitMarginals[3*6*6*6] = {};
    double stabMarginals[3*6*6*6] = {};

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

    double llr0 = log10(0.99/0.01);    //assuming 1% error chance
    for (int i=0; i<5*3*6*6*6; ++i) variableMessages[i] = llr0;

    for (int i=0; i<30; ++i)
    {
        updateFactorMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, variableMessages, 
                       syndrome, factorMessages, testCodeC.edgeToFaces, testCodeC.faceToEdges);
        updateVariableMessagesWrap(3*6*6*6, testCodeC.qubitInclusionLookup, variableMessages, 
                       factorMessages, testCodeC.faceToEdges, testCodeC.edgeToFaces, llr0);
    }
    calcMarginalsWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                       qubitMarginals, stabMarginals, factorMessages, llr0, llr0);
    bpCorrectionWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                      qubits, qubitMarginals, syndrome, stabMarginals, testCodeC.faceToEdges);

    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], 0);
        EXPECT_EQ(syndrome[i], 0);
    }
}
TEST(fullBPTest, MeasurementError)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    double variableMessages[5*3*6*6*6] = {};
    double factorMessages[5*3*6*6*6] = {};
    double qubitMarginals[3*6*6*6] = {};
    double stabMarginals[3*6*6*6] = {};

    syndrome[0] = 1;

    double llr0 = log10(0.99/0.01);    //assuming 1% error chance
    for (int i=0; i<5*3*6*6*6; ++i) variableMessages[i] = llr0;

    for (int i=0; i<30; ++i)
    {
        updateFactorMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, variableMessages, 
                       syndrome, factorMessages, testCodeC.edgeToFaces, testCodeC.faceToEdges);
        updateVariableMessagesWrap(3*6*6*6, testCodeC.qubitInclusionLookup, variableMessages, 
                       factorMessages, testCodeC.faceToEdges, testCodeC.edgeToFaces, llr0);
    }
    calcMarginalsWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                       qubitMarginals, stabMarginals, factorMessages, llr0, llr0);
    bpCorrectionWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                      qubits, qubitMarginals, syndrome, stabMarginals, testCodeC.faceToEdges);

    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], 0);
        EXPECT_EQ(syndrome[i], 0);
    }
}
TEST(fullBPTest, MeasurementAndQubitError)
{
    int qubits[3*6*6*6] = {};
    int syndrome[3*6*6*6] = {};
    double variableMessages[5*3*6*6*6] = {};
    double factorMessages[5*3*6*6*6] = {};
    double qubitMarginals[3*6*6*6] = {};
    double stabMarginals[3*6*6*6] = {};

    qubits[0] = 1;
    syndrome[0] = 1;
    syndrome[1] = 1;
    syndrome[4] = 1;

    double llr0 = log10(0.99/0.01);    //assuming 1% error chance
    for (int i=0; i<5*3*6*6*6; ++i) variableMessages[i] = llr0;

    for (int i=0; i<30; ++i)
    {
        updateFactorMessagesWrap(3*6*6*6, testCodeC.stabInclusionLookup, variableMessages, 
                       syndrome, factorMessages, testCodeC.edgeToFaces, testCodeC.faceToEdges);
        updateVariableMessagesWrap(3*6*6*6, testCodeC.qubitInclusionLookup, variableMessages, 
                       factorMessages, testCodeC.faceToEdges, testCodeC.edgeToFaces, llr0);
    }
    calcMarginalsWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                       qubitMarginals, stabMarginals, factorMessages, llr0, llr0);
    bpCorrectionWrap(3*6*6*6, testCodeC.qubitInclusionLookup, testCodeC.stabInclusionLookup, 
                      qubits, qubitMarginals, syndrome, stabMarginals, testCodeC.faceToEdges);

    for (int i=0; i<3*6*6*6; ++i)
    {
        EXPECT_EQ(qubits[i], 0);
        EXPECT_EQ(syndrome[i], 0);
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
