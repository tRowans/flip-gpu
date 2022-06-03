#include "codeObj.h"

//-----------------------PRIVATE-----------------------

int Code::coordToIndex(coord c)
{
    if (0 <= c.xi[0] && c.xi[0] < L &&  
        0 <= c.xi[1] && c.xi[1] < L && 
        0 <= c.xi[2] && c.xi[2] < L)
    {
        return (c.xi[0] + c.xi[1] * L + c.xi[2] * L * L);
    }
    else
    {
        throw std::invalid_argument("Coord out of range");
    }
}

coord Code::indexToCoord(int i)
{
    if (0 <= i && i < L*L*L)
    {
        coord c;
        c.xi[0] = i % L;
        c.xi[1] = (int)floor(i / L) % L;
        c.xi[2] = (int)floor(i / (L * L)) % L;
        return c;
    }
    else
    {
        throw std::invalid_argument("Index out of range");
    }
}

int Code::neigh(int v, int dir, int sign)
{
    coord c = indexToCoord(v, L);
    if (sign > 0)
    {
        if (dir == 0) c.xi[0] = (c.xi[0] + 1) % L;
        else if (dir == 1) c.xi[1] = (c.xi[1] + 1) % L;
        else if (dir == 2) c.xi[2] = (c.xi[2] + 1) % L;
    }
    else if (sign < 0)
    {
        if (dir == 0) c.xi[0] = (c.xi[0] - 1 + L) % L;
        else if (dir == 1) c.xi[1] = (c.xi[1] - 1 + L) % L;
        else if (dir == 2) c.xi[2] = (c.xi[2] - 1 + L) % L;
    }
    return coordToIndex(c, L);
}

int Code::edgeIndex(int v, int dir, int sign)
{
    if (sign < 0)
    {
        v = neigh(v, dir, sign, L);
    }
    if (dir == x)
        return 3 * v;
    else if (dir == y)
        return 3 * v + 1;
    else if (dir == z)
        return 3 * v + 2;
    else
        throw std::invalid_argument("Invalid direction");
}

void Code::buildFaceToEdges()
{
    for (int f=0; f<3*L*L*L; ++f)
    {
        int v = f / 3;
        int dir = f % 3;
        //The order we add them is important here 
        //Because this also defines the read schedule for the GPU code
        if (dir == 0)
        {
            faceToEdges[f][0] = edgeIndex(v, x, 1, L);
            faceToEdges[f][1] = edgeIndex(v, y, 1, L);
            faceToEdges[f][2] = edgeIndex(neigh(v, y, 1, L), x, 1, L);
            faceToEdges[f][3] = edgeIndex(neigh(v, x, 1, L), y, 1, L);
        }
        else if (dir == 1)
        {
            faceToEdges[f][0] = edgeIndex(v, z, 1, L);
            faceToEdges[f][1] = edgeIndex(v, x, 1, L);
            faceToEdges[f][2] = edgeIndex(neigh(v, x, 1, L), z, 1, L);
            faceToEdges[f][3] = edgeIndex(neigh(v, z, 1, L), x, 1, L);
        }
        else if (dir == 2)
        {
            faceToEdges[f][0] = edgeIndex(v, y, 1, L);
            faceToEdges[f][1] = edgeIndex(v, z, 1, L);
            faceToEdges[f][2] = edgeIndex(neigh(v, z, 1, L), y, 1, L);
            faceToEdges[f][3] = edgeIndex(neigh(v, y, 1, L), z, 1, L);
        }
    }
}

void Code::buildEdgeToFaces()
{
    for (int e=0; e<3*L*L*L; ++e)
    {
        int v = e / 3;
        int dir = e % 3;
        //Order important here as above
        if (dir == x)
        {
            edgeToFaces[e][0] = 3*v; //xy
            edgeToFaces[e][1] = 3*v + 1; // xz
            edgeToFaces[e][2] = 3*neigh(v, y, -1, L); // x,-y
            edgeToFaces[e][3] = 3*neigh(v, z, -1, L) + 1; // x,-z
        }
        else if (dir == y)
        {
            edgeToFaces[e][0] = 3*v + 2; // yz
            edgeToFaces[e][1] = 3 * v; // xy
            edgeToFaces[e][2] = 3*neigh(v, z, -1, L) + 2; // y,-z
            edgeToFaces[e][3] = 3*neigh(v, x, -1, L); // -x,y
        }
        else if (dir == z)
        {
            edgeToFaces[e][0] = 3*v + 1; // xz
            edgeToFaces[e][1] = 3*v + 2; // yz
            edgeToFaces[e][2] = 3*neigh(v, x, -1, L) + 1; // -x,z
            edgeToFaces[e][3] = 3*neigh(v, y, -1, L) + 2; // -y,z
        }
    }
}

/*
These are arrays with one element for every thread that will be used by a kernel
so that threads can check their value (0 or 1) to know if they correspond to a qubit/stab/logical or not
i.e. because they might correspond to a face that is out of bounds (if there are open boundaries)
or just because #threads per block * #blocks does not evenly divide #faces so there are spare threads
*/

void Code::buildQubitLookup()
{
    //((3*L*L*L+255)/256)*256 is the number of threads used in most kernels
    qubitInclusionLookup = new int[((3*L*L*L+255)/256)*256]();  //start all zeros
    
    //Open boundaries, logical Z runs top-to-bottom (z axis)
    if (bounds == 'o')
    {
        for (int f=0; f<3*L*L*L; ++f)
        {
            int v = f / 3;
            int dir = f % 3;
            coord cd = indexToCoord(v, L);
            if (cd.xi[0] < (L-3) && cd.xi[1] < (L-3) && cd.xi[2] < (L-3))
            {
                if (dir == 0) qubitInclusionLookup[f] = 1;
                else if (dir == 1 && cd.xi[1] > 0
                            && cd.xi[2] < L-4) qubitInclusionLookup[f] = 1;
                else if (dir == 2 && cd.xi[0] > 0
                            && cd.xi[2] < L-4) qubitInclusionLookup[f] = 1;
            }
        }
    }
    //Closed (periodic) boundaries
    else if (bounds == 'c')
    {
        for (int f=0; f<3*L*L*L; ++f) qubitInclusionLookup[f] = 1;
    }
    else throw std::invalid_argument("bounds must be \'o\' or \'c\'");
}

void Code::buildStabLookup()
{
    stabInclusionLookup = new int[((3*L*L*L+255)/256)*256]();  

    //Same boundaries as above
    if (bounds == 'o')
    {
        for (int e=0; e<3*L*L*L; e++)
        {
            int v = e / 3;
            int dir = e % 3;
            coord cd = indexToCoord(v, L);

            if (cd.xi[0] < (L-3) && cd.xi[1] < (L-3) && cd.xi[2] < (L-3))
            {
                if (dir == 0 && cd.xi[1] > 0) stabInclusionLookup[e] = 1;
                else if (dir == 1 && cd.xi[0] > 0) stabInclusionLookup[e] = 1;
                else if (dir == 2 && cd.xi[0] > 0
                                  && cd.xi[1] > 0
                                  && cd.xi[2] < (L-4)) stabInclusionLookup[e] = 1;
            }
        }
    }
    else if (bounds == 'c') 
    {
        for (int e=0; e<3*L*L*L; ++e) stabInclusionLookup[e] = 1;
    }
    else throw std::invalid_argument("bounds must be \'o\' or \'c\'");
}

void Code::buildLogicalLookup()
{
    //((3*L*L+63)/64)*64 threads used for this one
    logicalInclusionLookup = new int[((3*L*L+63)/64)*64]();
    if (bounds == 'o')
    {
        for (int y=0; y<L-3; ++y)
        {
            for (int x=0; x<L-3; ++x)
            {
                int baseQubit = 3*(x + y*L);
                logicalInclusionLookup[baseQubit] = 1;
            }
        }
    }
    else if (bounds == 'c')
    {
        for (int y=0; y<L; ++y)
        {
            for (int x=0; x<L; ++x)
            {
                int baseQubit = 3*(x + y*L);
                logicalInclusionLookup[baseQubit] = 1;
            }
        }
    }
    else throw std::invalid_argument("bounds must be \'o\' or \'c\'");
}


//-------------------------PUBLIC-------------------------

Code::Code(int L, char bounds)
{
    //This is a trick to get a dynamically allocated 3*L*L*L x 4 array in contiguous memory
    //which means the whole thing can easily be copied to the GPU
    //(3*L*L*L faces/edges x 4 edges/faces per face)
    faceToEdges = new int*[3*L*L*L]; //3*L*L*L pointers to arrays
    faceToEdges[0] = new int[3*L*L*L*4]; //point the first pointer to 3*L*L*L*4 ints 
    edgeToFaces = new int*[3*L*L*L];
    edgeToFaces[0] = new int[3*L*L*L*4];
    for (int i=1; i<3*L*L*L; ++i) 
    {
        faceToEdges[i] = faceToEdges[i-1] + 4; //point the rest to every 4th element
        edgeToFaces[i] = edgeToFaces[i-1] + 4;
    }

    buildFaceToEdges();
    buildEdgeToFaces();

    buildQubitLookup();
    buildStabLookup();
    buildLogicalLookup();
}

Code::~Code()
{
    delete[] faceToEdges[0];
    delete[] faceToEdges;
    delete[] edgeToFaces[0];
    delete[] edgeToFaces;
    delete[] qubitInclusionLookup;
    delete[] stabInclusionLookup;
    delete[] logicalInclusionLookup;
}
