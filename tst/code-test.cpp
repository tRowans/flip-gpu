#include "gtest/gtest.h"
#include "code.h"
#include "prepCode.h"

//------------------------------------------------------------

TEST(readParityCheckMatrixTest, surface)
{
    std::vector<std::vector<int>> H_Z;
    std::vector<std::vector<int>> H_X;
    H_Z = {{1,0,1,0},{0,1,0,1}};
    H_X = {{1,1,1,1}};
    EXPECT_EQ(H_Z, fourqsurface.H_Z);
    EXPECT_EQ(H_X, fourqsurface.H_X);
}
TEST(readParityCheckMatrixTest, colour)
{
    std::vector<std::vector<int>> H_Z;
    std::vector<std::vector<int>> H_X;
    H_Z = {{1,1,0,1,1,0,0},
           {0,1,1,0,1,1,0},
           {0,0,0,1,1,1,1}};
    H_X = {{1,1,0,1,1,0,0},
           {0,1,1,0,1,1,0},
           {0,0,0,1,1,1,1}};
    EXPECT_EQ(H_Z, sevenqcolour.H_Z);
    EXPECT_EQ(H_X, sevenqcolour.H_X);
}

//------------------------------------------------------------

TEST(findMaxBitDegreeTest, surface)
{
    EXPECT_EQ(fourqsurface.maxBitDegreeX, 1);
    EXPECT_EQ(fourqsurface.maxBitDegreeZ, 1);
}
TEST(findMaxBitDegreeTest, colour)
{
    EXPECT_EQ(sevenqcolour.maxBitDegreeX, 3);
    EXPECT_EQ(sevenqcolour.maxBitDegreeZ, 3);
}

//------------------------------------------------------------

TEST(findMaxCheckDegreeTest, surface)
{
    EXPECT_EQ(fourqsurface.maxCheckDegreeX, 4);
    EXPECT_EQ(fourqsurface.maxCheckDegreeZ, 2);
}
TEST(findMaxCheckDegreeTest, colour)
{
    EXPECT_EQ(sevenqcolour.maxCheckDegreeX, 4);
    EXPECT_EQ(sevenqcolour.maxCheckDegreeZ, 4);
}

//------------------------------------------------------------

TEST(buildBitToChecksTest, surface)
{
    int bitToXChecks[4][1] = {{0},{0},{0},{0}};
    int bitToZChecks[4][1] = {{0},{1},{0},{1}};
    for (int i=0; i<4; ++i)
    {
        EXPECT_EQ(fourqsurface.bitToXChecks[i][0], bitToXChecks[i][0]);
        EXPECT_EQ(fourqsurface.bitToZChecks[i][0], bitToZChecks[i][0]);
    }
}
TEST(buildBitToChecksTest, colour)
{
    int bitToXChecks[7][3] = {{0,-1,-1},
                              {0,1,-1},
                              {1,-1,-1},
                              {0,2,-1},
                              {0,1,2},
                              {1,2,-1},
                              {2,-1,-1}};
    int bitToZChecks[7][3] = {{0,-1,-1},
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
            EXPECT_EQ(sevenqcolour.bitToXChecks[i][j], bitToXChecks[i][j]);
            EXPECT_EQ(sevenqcolour.bitToZChecks[i][j], bitToZChecks[i][j]);
        }
    }
}

//------------------------------------------------------------

TEST(buildCheckToBitsTest, surface)
{
    int xCheckToBits[1][4] = {{0,1,2,3}};
    int zCheckToBits[2][2] = {{0,2},{1,3}};
    for (int i=0; i<4; ++i) EXPECT_EQ(fourqsurface.xCheckToBits[0][i], xCheckToBits[0][i]);
    for (int i=0; i<2; ++i)
    {
        for (int j=0; j<2; ++j) EXPECT_EQ(fourqsurface.zCheckToBits[i][j], zCheckToBits[i][j]);
    }
}
TEST(buildCheckToBitsTest, colour)
{
    int xCheckToBits[3][4] = {{0,1,3,4},
                              {1,2,4,5},
                              {3,4,5,6}};
    int zCheckToBits[3][4] = {{0,1,3,4},
                              {1,2,4,5},
                              {3,4,5,6}};
    for (int i=0; i<3; ++i)
    {
        for (int j=0; j<4; ++j)
        {
            EXPECT_EQ(sevenqcolour.xCheckToBits[i][j], xCheckToBits[i][j]);
            EXPECT_EQ(sevenqcolour.zCheckToBits[i][j], zCheckToBits[i][j]);
        }
    }
}
