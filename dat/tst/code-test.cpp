#include "gtest/gtest.h"
#include "code.h"
#include "prepCode.h"

//------------------------------------------------------------
    
TEST(coordToIndexTest, HandlesExpectedInput)
{
    coord c = {4,4,1};
    EXPECT_EQ(testCodeC.coordToIndex(c), 64);
} 
TEST(coordToIndexTest, HandlesOutOfRangeInput)
{
    coord c = {-1,0,0};
    EXPECT_THROW(testCodeC.coordToIndex(c), std::invalid_argument);
    c.xi[0] = 6;
    EXPECT_THROW(testCodeC.coordToIndex(c), std::invalid_argument);
}

//------------------------------------------------------------

TEST(indexToCoordTest, HandlesExpectedInput)
{
    coord c = testCodeC.indexToCoord(64);
    EXPECT_EQ(c.xi[0], 4);
    EXPECT_EQ(c.xi[1], 4);
    EXPECT_EQ(c.xi[2], 1);
}
TEST(indexToCoordTest, HandlesOutOfRangeInput)
{
    EXPECT_THROW(testCodeC.indexToCoord(-1), std::invalid_argument);
    EXPECT_THROW(testCodeC.indexToCoord(216), std::invalid_argument);
}

//------------------------------------------------------------

TEST(neighTest, CorrectOutput)
{
    //129 = {3,3,3}
    EXPECT_EQ(testCodeC.neigh(129, 0, 1), 130);
    EXPECT_EQ(testCodeC.neigh(129, 1, 1), 135);
    EXPECT_EQ(testCodeC.neigh(129, 2, 1), 165);
    EXPECT_EQ(testCodeC.neigh(129, 0, -1), 128);
    EXPECT_EQ(testCodeC.neigh(129, 1, -1), 123);
    EXPECT_EQ(testCodeC.neigh(129, 2, -1), 93);
}

//------------------------------------------------------------

TEST(edgeIndexTest, HandlesExpectedInput)
{
    EXPECT_EQ(testCodeC.edgeIndex(129, 0, 1), 387);
    EXPECT_EQ(testCodeC.edgeIndex(129, 1, 1), 388);
    EXPECT_EQ(testCodeC.edgeIndex(129, 2, 1), 389);
    EXPECT_EQ(testCodeC.edgeIndex(129, 0, -1), 384);
    EXPECT_EQ(testCodeC.edgeIndex(129, 1, -1), 370);
    EXPECT_EQ(testCodeC.edgeIndex(129, 2, -1), 281);
}
TEST(edgeIndexTest, HandlesInvalidInput)
{
    EXPECT_THROW(testCodeC.edgeIndex(0, 3, 1), std::invalid_argument);
}

//------------------------------------------------------------

TEST(buildFaceToEdgesTest, CorrectOutput)
{   
    //Not worth checking the whole thing
    //Just check this for the three directions of face and check size is right
    //Same for other functions of this type
    int edges1[4] = {387, 388, 405, 391};
    int edges2[4] = {389, 387, 392, 495};
    int edges3[4] = {388, 389, 496, 407};
    for (int i=0; i<4; ++i)
    {
        EXPECT_EQ(testCodeC.faceToEdges[387][i], edges1[i]);
        EXPECT_EQ(testCodeC.faceToEdges[388][i], edges2[i]);
        EXPECT_EQ(testCodeC.faceToEdges[389][i], edges3[i]);
    }
}

//------------------------------------------------------------

TEST(buildEdgeToFacesTest, CorrectOutput)
{
    int faces1[4] = {387, 388, 369, 280};
    int faces2[4] = {389, 387, 281, 384};
    int faces3[4] = {388, 389, 385, 371};
    for (int i=0; i<4; ++i)
    {
        EXPECT_EQ(testCodeC.edgeToFaces[387][i], faces1[i]);
        EXPECT_EQ(testCodeC.edgeToFaces[388][i], faces2[i]);
        EXPECT_EQ(testCodeC.edgeToFaces[389][i], faces3[i]); 
    }
}

//------------------------------------------------------------

TEST(buildQubitLookupTest, ClosedBoundaries)
{
    int* lookupExpected = new int[((3*6*6*6+255)/256)*256]();
    for (int i=0; i<3*6*6*6; ++i)
    {
        lookupExpected[i] = 1;
    }
    for (int i=0; i<((3*6*6*6+255)/256)*256; ++i)
    {
        EXPECT_EQ(testCodeC.qubitInclusionLookup[i], lookupExpected[i]);
    }
    delete[] lookupExpected;
}
TEST(buildQubitLookupTest, OpenBoundaries)
{
    int* lookupExpected = new int[((3*6*6*6+255)/256)*256]();
    int includedQubits[51] = {0,3,5,6,8,18,19,21,22,23,24,25,26,36,37,39,40,41,42,43,44,
                              108,111,113,114,116,126,127,129,130,131,132,133,134,144,145,147,148,149,150,151,152,
                              216,219,222,234,237,240,252,255,258};
    for (int i=0; i<51; ++i)
    {
        lookupExpected[includedQubits[i]] = 1;
    }
    for (int i=0; i<((3*6*6*6+255)/256)*256; ++i)
    {
        EXPECT_EQ(testCodeO.qubitInclusionLookup[i], lookupExpected[i]);
    }
    delete[] lookupExpected;
}

//------------------------------------------------------------

TEST(buildStabLookupTest, ClosedBoundaries)
{
    int* lookupExpected = new int[((3*6*6*6+255)/256)*256]();
    for (int i=0; i<3*6*6*6; ++i)
    {
        lookupExpected[i] = 1;
    }
    for (int i=0; i<((3*6*6*6+255)/256)*256; ++i)
    {
        EXPECT_EQ(testCodeC.stabInclusionLookup[i], lookupExpected[i]);
    }
    delete[] lookupExpected;
}
TEST(buildStabLookupTest, OpenBoundaries)
{
    
    int* lookupExpected = new int[((3*6*6*6+255)/256)*256]();
    int includedStabs[44] = {4,7,18,21,22,23,24,25,26,36,39,40,41,42,43,44,
                             112,115,126,129,130,131,132,133,134,144,147,148,149,150,151,152,
                             220,223,234,237,238,240,241,252,255,256,258,259};
    for (int i=0; i<44; ++i)
    {
        lookupExpected[includedStabs[i]] = 1;
    }
    for (int i=0; i<((3*6*6*6+255)/256)*256; ++i)
    {
        EXPECT_EQ(testCodeO.stabInclusionLookup[i], lookupExpected[i]);
    }
    delete[] lookupExpected;
}

//------------------------------------------------------------

TEST(buildLogicalLookupTest, ClosedBoundaries)
{
    int* lookupExpected = new int[((3*6*6+63)/64)*64]();
    int includedLogicals[36] = {0,3,6,9,12,15,
                               18,21,24,27,30,33,
                               36,39,42,45,48,51,
                               54,57,60,63,66,69,
                               72,75,78,81,84,87,
                               90,93,96,99,102,105};
    for (int i=0; i<36; ++i)
    {
        lookupExpected[includedLogicals[i]] = 1;
    }
    for (int i=0; i<((3*6*6+63)/64)*64; ++i)
    {
        EXPECT_EQ(testCodeC.logicalInclusionLookup[i], lookupExpected[i]);
    }
    delete[] lookupExpected;
}
TEST(buildLogicalLookupTest, OpenBoundaries)
{
    int* lookupExpected = new int[((3*6*6+63)/64)*64]();
    int includedLogicals[9] = {0,3,6,
                               18,21,24,
                               36,39,42};
    for (int i=0; i<9; ++i)
    {
        lookupExpected[includedLogicals[i]] = 1;
    }
    for (int i=0; i<((3*6*6+63)/64)*64; ++i)
    {
        EXPECT_EQ(testCodeO.logicalInclusionLookup[i], lookupExpected[i]);
    }
    delete[] lookupExpected;
}
