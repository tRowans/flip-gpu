#include "gtest/gtest.h"
#include "code.h"
#include "prepCode.h"

//------------------------------------------------------------
    
TEST(coordToIndexTest, HandlesExpectedInput)
{
    coord c = {4,4,1};
    EXPECT_EQ(coordToIndex(c, 6), 64);
} 
TEST(coordToIndexTest, HandlesOutOfRangeInput)
{
    coord c = {-1,0,0};
    EXPECT_THROW(coordToIndex(c, 6), std::invalid_argument);
    c.xi[0] = 6;
    EXPECT_THROW(coordToIndex(c, 6), std::invalid_argument);
}

//------------------------------------------------------------

TEST(indexToCoordTest, HandlesExpectedInput)
{
    coord c = indexToCoord(64, 6);
    EXPECT_EQ(c.xi[0], 4);
    EXPECT_EQ(c.xi[1], 4);
    EXPECT_EQ(c.xi[2], 1);
}
TEST(indexToCoordTest, HandlesOutOfRangeInput)
{
    EXPECT_THROW(indexToCoord(-1, 6), std::invalid_argument);
    EXPECT_THROW(indexToCoord(216, 6), std::invalid_argument);
}

//------------------------------------------------------------

TEST(neighTest, CorrectOutput)
{
    //129 = {3,3,3}
    EXPECT_EQ(neigh(129, 0, 1, 6), 130);
    EXPECT_EQ(neigh(129, 1, 1, 6), 135);
    EXPECT_EQ(neigh(129, 2, 1, 6), 165);
    EXPECT_EQ(neigh(129, 0, -1, 6), 128);
    EXPECT_EQ(neigh(129, 1, -1, 6), 123);
    EXPECT_EQ(neigh(129, 2, -1, 6), 93);
}

//------------------------------------------------------------

TEST(edgeIndexTest, HandlesExpectedInput)
{
    EXPECT_EQ(edgeIndex(129, 0, 1, 6), 387);
    EXPECT_EQ(edgeIndex(129, 1, 1, 6), 388);
    EXPECT_EQ(edgeIndex(129, 2, 1, 6), 389);
    EXPECT_EQ(edgeIndex(129, 0, -1, 6), 384);
    EXPECT_EQ(edgeIndex(129, 1, -1, 6), 370);
    EXPECT_EQ(edgeIndex(129, 2, -1, 6), 281);
}
TEST(edgeIndexTest, HandlesInvalidInput)
{
    EXPECT_THROW(edgeIndex(0, 3, 1, 5), std::invalid_argument);
}

//------------------------------------------------------------

TEST(buildFaceToEdgesTest, CorrectOutput)
{   
    //Not worth checking the whole thing
    //Just check this for the three directions of face and check size is right
    //Same for other functions of this type
    vvint faceToEdges; 
    buildFaceToEdges(faceToEdges, 6);
    vint edges1 = {387, 388, 391, 405};
    vint edges2 = {387, 389, 392, 495};
    vint edges3 = {388, 389, 407, 496};
    EXPECT_EQ(faceToEdges[387], edges1);
    EXPECT_EQ(faceToEdges[388], edges2);
    EXPECT_EQ(faceToEdges[389], edges3);
    EXPECT_EQ(faceToEdges.size(), 3*6*6*6);
}

//------------------------------------------------------------

TEST(buildEdgeToFaces, CorrectOutput)
{
    vvint edgeToFaces;
    buildEdgeToFaces(edgeToFaces, 6);
    vint faces1 = {387, 388, 369, 280};
    vint faces2 = {387, 389, 384, 281};
    vint faces3 = {389, 388, 371, 385};
    EXPECT_EQ(edgeToFaces[387], faces1);
    EXPECT_EQ(edgeToFaces[388], faces2);
    EXPECT_EQ(edgeToFaces[389], faces3); 
    EXPECT_EQ(edgeToFaces.size(), 3*6*6*6); 
}

//------------------------------------------------------------

TEST(buildVertexToEdgesTest, CorrectOutput)
{
    vvint vertexToEdges;
    buildVertexToEdges(vertexToEdges, 6);
    vint edges = {387, 388, 389, 384, 370, 281};
    EXPECT_EQ(vertexToEdges[129], edges);
    EXPECT_EQ(vertexToEdges.size(), 6*6*6);
}

//------------------------------------------------------------

TEST(buildEdgeToVerticesTest, CorrectOutput)
{
    vpint edgeToVertices;
    buildEdgeToVertices(edgeToVertices, 6);
    std::pair<int,int> vertices1 = {129, 130};
    std::pair<int,int> vertices2 = {129, 135};
    std::pair<int,int> vertices3 = {129, 165};
    EXPECT_EQ(edgeToVertices[387], vertices1);
    EXPECT_EQ(edgeToVertices[388], vertices2);
    EXPECT_EQ(edgeToVertices[389], vertices3);
    EXPECT_EQ(edgeToVertices.size(), 3*6*6*6);
}

//------------------------------------------------------------

TEST(buildFaceToVerticesTest, CorrectOutput)
{
    vvint faceToVertices;
    buildFaceToVertices(faceToVertices, 6);
    vint vertices1 = {129, 130, 135, 136};
    vint vertices2 = {129, 130, 165, 166};
    vint vertices3 = {129, 135, 165, 171};
    EXPECT_EQ(faceToVertices[387], vertices1);
    EXPECT_EQ(faceToVertices[388], vertices2);
    EXPECT_EQ(faceToVertices[389], vertices3);
    EXPECT_EQ(faceToVertices.size(), 3*6*6*6);
}

//------------------------------------------------------------

TEST(buildXLogicalsTest, CorrectOutput)
{
    vvint zLogicals;
    buildZLogicals(zLogicals, 3);
    vvint zLogicalsExpected;
    zLogicalsExpected.push_back({2,5,8});
    zLogicalsExpected.push_back({11,14,17});
    zLogicalsExpected.push_back({20,23,26});
    zLogicalsExpected.push_back({29,32,35});
    zLogicalsExpected.push_back({38,41,44});
    zLogicalsExpected.push_back({47,50,53});
    zLogicalsExpected.push_back({56,59,62});
    zLogicalsExpected.push_back({65,68,71});
    zLogicalsExpected.push_back({74,77,80});
    EXPECT_EQ(zLogicals, zLogicalsExpected);
}

//------------------------------------------------------------

TEST(relaxEqualTest, ReduceSyndromeWeight)
{
    std::random_device rd{};
    std::mt19937 engine{rd()};
    std::uniform_real_distribution<double> dist(0,1);

    buildLattice(latCubic);
    latCubic.wipe();
    latCubic.qubits[0] = 1;
    latCubic.calcSynd();
    relaxEqual(latCubic, 1, engine, dist);

    vint qubitsExpected(3*6*6*6, 0);
    EXPECT_EQ(latCubic.qubits, qubitsExpected);
}
TEST(relaxEqualTest, PreseveSyndromeWeight)
{
    std::random_device rd{};
    std::mt19937 engine{rd()};
    std::uniform_real_distribution<double> dist(0,1);

    buildLattice(latCubic);
    latCubic.wipe();
    latCubic.qubits[2] = 1;
    latCubic.qubits[20] = 1;
    latCubic.qubits[110] = 1;
    latCubic.qubits[128] = 1;
    latCubic.calcSynd();

    vint qubitsExpected(3*6*6*6, 0);
    qubitsExpected[2] = 1;
    qubitsExpected[20] = 1;
    qubitsExpected[110] = 1;
    qubitsExpected[128] = 1;

    relaxEqual(latCubic, 1, engine, dist);
    for (int i = 0; i < 3*6*6*6; i++)
    {
        if (i != 2 && i != 20 && i != 110 && i != 128)
        {
            EXPECT_EQ(latCubic.qubits[i], 0);
        }
    }
}

