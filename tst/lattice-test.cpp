#include "gtest/gtest.h"
#include "cubic.h"
#include "prepLattice.h"

//------------------------------------------------------------

TEST(indexBuilding, build)
{
    //Most of the functions here need populated index vectors to work
    //The lattice building functions are tested separately in cubic-test.cpp
    buildLattice(lattice);   
    buildLattice(lattice3);
    ASSERT_TRUE(true);
}

//------------------------------------------------------------

TEST(constructorTest, CheckSetValues)
{
    Lattice lattice(6);
    vint emptyVector = {};
    EXPECT_EQ(lattice.L,6);
    EXPECT_EQ(lattice.qubits.size(),3*6*6*6);
    EXPECT_EQ(lattice.syndrome.size(),3*6*6*6);
}

//------------------------------------------------------------

TEST(wipeTest, VectorsClear)
{
    for (int i = 0; i < lattice.qubits.size(); i++) lattice.qubits[i] = 1;
    for (int i = 0; i < lattice.syndrome.size(); i++) lattice.syndrome[i] = 1;
    lattice.wipe();
    for (int i : lattice.qubits) {if (i != 0) FAIL();}
    for (int i : lattice.syndrome) {if (i != 0) FAIL();}
}

//------------------------------------------------------------

//Not sure about the best way to test the random functions
//Maybe just test that they do something, if not a specific error
//Could do nothing but chance is very small

TEST(qubitErrorTest, NonTrivialAction)
{
    lattice.wipe();

    std::random_device rd{};
    std::mt19937 engine{rd()};
    std::uniform_real_distribution<double> dist(0,1);

    lattice.qubitError(0.5,engine,dist);
    int nonZeroCheck = 0;
    for (int i : lattice.qubits) {if (i != 0) nonZeroCheck = 1;}
    EXPECT_EQ(nonZeroCheck, 1);
}

//------------------------------------------------------------

TEST(measErrorTest, NonTrivialAction)
{
    lattice.wipe();

    std::random_device rd{};
    std::mt19937 engine{rd()};
    std::uniform_real_distribution<double> dist(0,1);

    lattice.measError(0.5, engine, dist);
    int nonZeroCheck = 0;
    for (int i : lattice.syndrome) {if (i != 0) nonZeroCheck = 1;}
    EXPECT_EQ(nonZeroCheck, 1);
}

//------------------------------------------------------------

TEST(calcSyndTest, CorrectOutputOneError)
{
    lattice.wipe();
    lattice.qubits[20] = 1;
    lattice.calcSynd();
    vint syndromeExpected(3*6*6*6, 0);
    syndromeExpected[19] = 1;
    syndromeExpected[20] = 1;
    syndromeExpected[38] = 1;
    syndromeExpected[127] = 1;
    EXPECT_EQ(lattice.syndrome, syndromeExpected);
}
TEST(calcSyndTest, CorrectOutputTwoErrors)
{
    lattice.wipe();
    lattice.qubits[20] = 1;
    lattice.qubits[128] = 1;
    lattice.calcSynd();
    vint syndromeExpected(3*6*6*6, 0);
    syndromeExpected[19] = 1;
    syndromeExpected[20] = 1;
    syndromeExpected[38] = 1;
    syndromeExpected[128] = 1;
    syndromeExpected[146] = 1;
    syndromeExpected[235] = 1;
    EXPECT_EQ(lattice.syndrome, syndromeExpected);
}

//------------------------------------------------------------

TEST(checkLogicalErrorTest, NoError)
{
    lattice.wipe();
    EXPECT_EQ(lattice.checkLogicalError(), 0);
}
TEST(checkLogicalErrorTest, CorrectableXError)
{
    lattice3.wipe();
    lattice3.qubits[2] = 1;
    lattice3.qubits[11] = 1;
    lattice3.qubits[20] = 1;
    lattice3.qubits[29] = 1;
    EXPECT_EQ(lattice3.checkLogicalError(), 0);
}
TEST(checkLogicalErrorTest, LogicalXError)
{
    lattice3.wipe();
    lattice3.qubits[2] = 1;
    lattice3.qubits[11] = 1;
    lattice3.qubits[20] = 1;
    lattice3.qubits[29] = 1;
    lattice3.qubits[38] = 1;
    lattice3.qubits[47] = 1;
    lattice3.qubits[56] = 1;
    lattice3.qubits[65] = 1;
    lattice3.qubits[74] = 1;
    EXPECT_EQ(lattice3.checkLogicalError(), 1);
}
TEST(checkLogicalErrorTest, UncorrectableXError)
{
    lattice3.wipe();
    lattice3.qubits[2] = 1;
    lattice3.qubits[11] = 1;
    lattice3.qubits[20] = 1;
    lattice3.qubits[29] = 1;
    lattice3.qubits[38] = 1;
    EXPECT_EQ(lattice3.checkLogicalError(), 1);
}

