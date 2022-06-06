#include<random>
#include<iostream>
#include "decode.cuh"

int main(int argc, char *argv[])
{
    if (argc != 9)
    {
        std::cout << "Invalid number of arguments." << '\n';
        return 1;
    }

    int L = std::atoi(argv[1]);         //Lattice size
    double p = std::atof(argv[2]);      //qubit error prob
    double q = std::atof(argv[3]);      //meas error prob
    int runs = std::atoi(argv[4]);      //sim repeats
    int cycles = std::atoi(argv[5]);    //code cycles per sim
    int apps = std::atoi(argv[6]);      //applications of flip per code cycle
    int pfreq = std::atoi(argv[7]);     //apply pFlip instead of flip every pfreq applications
    char bounds = std::atoi(argv[8]);   //open ('o') or closed ('c') boundary conditions

    int N = 3*L*L*L; //number of lattice faces/edges (= number of qubits/edges if there are no boundaries)
   
    //build code info 
    Code code(L, bounds);

    //pointers for arrays on device
    int *d_qubits, *d_syndrome;
    int *d_faceToEdges, *d_edgeToFaces;
    int *d_qubitInclusionLookup, *d_stabInclusionLookup, *d_logicalInclusionLookup;

    //don't need to copy for these, just set to all zeros on device (later)
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, N*sizeof(int));

    //these get copied to device from initialised versions in code object
    cudaMalloc(&d_faceToEdges, 4*N*sizeof(int));
    cudaMemcpy(d_faceToEdges, code.faceToEdges[0], 
                4*N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_edgeToFaces, 4*N*sizeof(int));
    cudaMemcpy(d_edgeToFaces, code.edgeToFaces[0], 
                4*N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_qubitInclusionLookup, ((N+255)/256)*256*sizeof(int));
    cudaMemcpy(d_qubitInclusionLookup, code.qubitInclusionLookup,
                ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_stabInclusionLookup, ((N+255)/256)*256*sizeof(int));
    cudaMemcpy(d_stabInclusionLookup, code.stabInclusionLookup,
                ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_logicalInclusionLookup, ((N+255)/256)*256*sizeof(int));
    cudaMemcpy(d_logicalInclusionLookup, code.logicalInclusionLookup,
                ((N+255)/256)*256*sizeof(int), cudaMemcpyHostToDevice);
        
    int failures = 0;       //count for total logical errors
    int* nOdd = 0;          //count for number of -1 logicals in majority vote
    cudaMallocManaged(&nOdd, sizeof(int));  //can be accessed by cpu or gpu

    //setup state array for device-side random number generation
    std::random_device rd{};
    curandState_t *d_states;
    createStates<<<(N+255)/256,256>>>(N, rd(), d_states);
    
    for (int run=0; run<runs; ++run)
    {
        //set qubits and syndrome to all zeros 
        wipeArrays<<<(N+255)/256,256>>>(N, d_qubits, d_syndrome);

        for (int cycle=0; cycle<cycles; ++cycle)
        {
            applyErrors<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_states, d_qubits, p);                   //qubit errors
            updateSyndrome<<<(N+255)/256,256>>>(d_stabInclusionLookup, d_qubits, d_syndrome, d_edgeToFaces);   //syndrome measurement
            applyErrors<<<(N+255)/256,256>>>(d_stabInclusionLookup, d_states, d_syndrome, q);                  //measurement errors
            for (int app=1; app<apps+1; ++app)  //start on 1 instead of 0 so app % pfreq != 0 on the first loop (unless pfreq=1)
            {
                //use pflip every pfreq applications, otherwise use regular flip
                if (app % pfreq == 0) pflip<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_qubits, 
                                                                    d_syndrome, d_faceToEdges, d_states);
                else flip<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_qubits, d_syndrome, d_faceToEdges);
            }
        }

        *nOdd = 0;
        //measure parity of all disjoint logical Z reps
        measureLogicals<<<(L*L+63)/64,64>>>(d_logicalInclusionLookup, d_qubits, nOdd, L, bounds);  
        if (*nOdd >= (L*L)/2) failures += 1;  //majority vote (L*L = number of disjoint logical Z reps)
    }

    std::cout << L << ',' << p << ',' << q << ',' << runs << ',' << cycles << ',' << apps << ',' << pfreq << ',' << failures << '\n';
    return 0;
}
