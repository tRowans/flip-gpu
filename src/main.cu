#include<random>
#include<iostream>
#include "code.h"
#include "decode.cuh"

int main(int argc, char *argv[])
{
    if (argc != 11)
    {
        std::cout << "Invalid number of arguments." << '\n';
        return 1;
    }

    int L = std::atoi(argv[1]);          //lattice size
    double pLower = std::atof(argv[2]);  //lower value for error probability p
    double pUpper = std::atof(argv[3]);  //upper value for p
    int nps = std::atoi(argv[4]);     //number of values for p in range pLower <= p <= pUpper
    double alpha = std::atof(argv[5]);   //measurement error probability q = alpha*p
    int runs = std::atoi(argv[6]);       //number of repeats of simulation
    int cycles = std::atoi(argv[7]);     //code cycles per simulation
    int apps = std::atoi(argv[8]);       //applications of decoding rule per code cycle
    int pfreq = std::atoi(argv[9]);      //apply pFlip instead of flip every pfreq applications
    char bounds = *argv[10];             //open ('o') or closed ('c') boundary conditions

    int N = 3*L*L*L; //number of lattice faces/edges (= number of qubits/edges if there are no boundaries)

    double pRange = pUpper - pLower;
    double pStep;
    if (nps == 1) pStep = 0;
    else pStep = pRange/(nps-1);
    double ps[nps];
    double qs[nps];
    for (int i=0; i<nps; ++i)
    {
        ps[i] = pLower + i*pStep;
        qs[i] = alpha*ps[i];
    }
      
    //build code info 
    Code code(L, bounds);

    //pointers for arrays on device
    int *d_qubits, *d_syndrome;
    float *d_qubitMessages, *d_syndromeMessages, *d_qubitMarginals;
    int *d_faceToEdges, *d_edgeToFaces;
    int *d_qubitInclusionLookup, *d_stabInclusionLookup, *d_logicalInclusionLookup;

    //don't need to copy for these, just set to all zeros on device (later)
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, N*sizeof(int));
    cudaMalloc(&d_qubitMessages, 8*N*sizeof(float));
    cudaMalloc(&d_syndromeMessages, 8*N*sizeof(float));
    cudaMalloc(&d_qubitMarginals, N*sizeof(float));

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
    
    cudaMalloc(&d_logicalInclusionLookup, ((3*L*L+63)/64)*64*sizeof(int));
    cudaMemcpy(d_logicalInclusionLookup, code.logicalInclusionLookup,
                ((3*L*L+63)/64)*64*sizeof(int), cudaMemcpyHostToDevice);
        
    int failures[nps] = {};                 //count for total logical errors per value of p
    int* nOdd;                              //count for number of -1 logicals in majority vote
    cudaMallocManaged(&nOdd, sizeof(int));  //can be accessed by cpu or gpu

    //setup state array for device-side random number generation
    std::random_device rd{};
    curandState_t *d_states;
    cudaMalloc(&d_states, N*sizeof(curandState_t));
    createStates<<<(N+255)/256,256>>>(N, rd(), d_states);
    cudaDeviceSynchronize();

    for (int i=0; i<nps; ++i)
    {
        for (int run=0; run<runs; ++run)
        {
            //set qubits and syndrome to all zeros 
            wipeArray<<<(N+255)/256,256>>>(N, d_qubits);
            cudaDeviceSynchronize();
        
            for (int cycle=1; cycle<cycles+1; ++cycle) //starts on 1 instead of 0 so cycle % pfreq != 0 on the first loop
            {
                applyErrors<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_states, d_qubits, ps[i]);                   //qubit errors
                cudaDeviceSynchronize();
                calculateSyndrome<<<(N+255)/256,256>>>(d_stabInclusionLookup, d_qubits, d_syndrome, d_edgeToFaces);    //measure stabilisers
                cudaDeviceSynchronize();
                applyErrors<<<(N+255)/256,256>>>(d_stabInclusionLookup, d_states, d_syndrome, qs[i]);                  //measurement errors
                cudaDeviceSynchronize();
                        
                for (int app=1; app<apps; ++app)  //start on 1 instead of 0 so app % pfreq != 0 on the first loop (unless pfreq=1)
                {
                    //use pflip every pfreq applications, otherwise use regular flip
                    updateSyndromeMessages<<<(N+255)/256,256>>>(d_stabInclusionLookup, d_qubitMessages, d_syndrome, 
                                                                    d_syndromeMessages, d_edgeToFaces, d_faceToEdges);
                    cudaDeviceSynchronize();
                    updateQubitMessages<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_qubitMessages, d_syndromeMessages, 
                                                                                    d_faceToEdges, d_edgeToFaces, ps[i]);
                    calcMarginals<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_qubitMarginals, d_syndromeMessages, ps[i]);
                    cudaDeviceSynchronize();
                    if (app % pfreq == 0) pflip<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_stabInclusionLookup, 
                                                                           d_qubits, d_syndrome, d_faceToEdges, d_edgeToFaces, 
                                                                           d_qubitMessages, d_qubitMarginals, d_states);
                    else flip<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_stabInclusionLookup, 
                                                          d_qubits, d_syndrome, d_faceToEdges, d_edgeToFaces,
                                                          d_qubitMessages, d_qubitMarginals);
                    cudaDeviceSynchronize();
                }

                //Flip all qubits with marginals > 0.5 for final iteration instead of using flip
                updateSyndromeMessages<<<(N+255)/256,256>>>(d_stabInclusionLookup, d_qubitMessages, d_syndrome, 
                                                                    d_syndromeMessages, d_edgeToFaces, d_faceToEdges);
                cudaDeviceSynchronize();
                updateQubitMessages<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_qubitMessages, d_syndromeMessages, 
                                                                                d_faceToEdges, d_edgeToFaces, ps[i]);
                calcMarginals<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_qubitMarginals, d_syndromeMessages, ps[i]);
                cudaDeviceSynchronize();
                bpCorrection<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_qubits, d_qubitMarginals);
                cudaDeviceSynchronize();
            }

            *nOdd = 0;
            //measure parity of all disjoint logical Z reps
            measureLogicals<<<(3*L*L+63)/64,64>>>(d_logicalInclusionLookup, d_qubits, nOdd, L, bounds);  
            cudaDeviceSynchronize();
            int nZReps = 0;
            for (int j=0; j<((3*L*L+63)/64)*64; ++j) nZReps += code.logicalInclusionLookup[j];
            if (*nOdd >= nZReps/2) failures[i] += 1;  //majority vote (L*L = number of disjoint logical Z reps)
        }

        std::cout << L << ',' << ps[i] << ',' << qs[i] << ',' << runs << ',' << failures[i] << '\n';

    }

    cudaFree(d_qubits);
    cudaFree(d_syndrome);
    cudaFree(d_faceToEdges);
    cudaFree(d_edgeToFaces);
    cudaFree(d_qubitInclusionLookup);
    cudaFree(d_stabInclusionLookup);
    cudaFree(d_logicalInclusionLookup);
    cudaFree(nOdd);
    return 0;
}
