#ifndef CODE_H
#define CODE_H

#include<stdexcept>

struct coord
{
    int xi[3];
};

enum direction { x, y, z, };

//Basically just a neat container for some data
class Code
{
    public:
        //params
        int L;          //lattice size
        char bounds;    //open or closed boundaries
        
        //data vector setup functions
        //these are only public so that they can be accessed in unit tests
        //apart from that they are not used outside of object construction
        int coordToIndex(coord c);
        coord indexToCoord(int i);
        int neigh(int v, int dir, int sign);
        int edgeIndex(int v, int dir, int sign);
        void buildFaceToEdges();
        void buildEdgeToFaces();
        void buildQubitLookup();
        void buildStabLookup();
        void buildLogicalLookup();

        //data vectors
        int** faceToEdges;      //cuda does not like copying to or from vectors etc 
        int** edgeToFaces;      //so these need to be set up like this
        int* qubitInclusionLookup;   //(the vectors above do not get used by the gpu)
        int* stabInclusionLookup;
        int* logicalInclusionLookup; 

        Code(int lIn, char boundsIn);
        ~Code();
};

#endif
