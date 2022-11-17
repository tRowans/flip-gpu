#include<random>
#include<iostream>
#include<fstream>
#include<string>
#include<math.h>
#include "code.h"
#include "decode.cuh"

int main(int argc, char *argv[])
{
    if (argc != 13)
    {
        std::cout << "Error: invalid number of arguments\n";
    }
    
    int L = std::atoi(argv[1]);          //lattice size
    double pLower = std::atof(argv[2]);  //lower value for error probability p
    double pUpper = std::atof(argv[3]);  //upper value for error probability p
    int nps = std::atoi(argv[4]);        //number of values for p in range pLower <= p <= pUpper
    double alpha = std::atof(argv[5]);   //measurement error probability q = alpha*p
    int runs = std::atoi(argv[6]);       //number of repeats of simulation
    int cycles = std::atoi(argv[7]);     //code cycles per simulation
    int bpIters = std::atoi(argv[8]);    //BP iterations per code cycle
    int useFlip = std::atoi(argv[9]);    //use flip in decoding? (0=pure BP, 1=hybrid BP-flip)
    int flipIters = std:atoi(argv[10]);  //flip iterations per code cycle
    int pfreq = std::atoi(argv[11]);     //apply pFlip instead of flip every pfreq applications
    char bounds = *argv[12];             //open ('o') or closed ('c') boundary conditions

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
    double *d_variableMessages, *d_factorMessages, *d_qubitMarginals, *d_stabMarginals;
    int *d_faceToEdges, *d_edgeToFaces;
    int *d_qubitInclusionLookup, *d_stabInclusionLookup, *d_logicalInclusionLookup;

    //don't need to copy for these, just set to all zeros on device (later)
    cudaMalloc(&d_qubits, N*sizeof(int));
    cudaMalloc(&d_syndrome, N*sizeof(int));
    cudaMalloc(&d_variableMessages, 5*N*sizeof(double));
    cudaMalloc(&d_factorMessages, 5*N*sizeof(double));
    cudaMalloc(&d_qubitMarginals, N*sizeof(double));
    cudaMalloc(&d_stabMarginals, N*sizeof(double));

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
        double llr0 = log10((1-ps[i])/ps[i]);
        double llrq0 = log10((1-qs[i])/qs[i]);
        for (int run=0; run<runs; ++run)
        {
            //set qubits and syndrome to all zeros 
            wipeArray<<<(N+255)/256,256>>>(N, d_qubits);
            cudaDeviceSynchronize();
        
            for (int cycle=0; cycle<cycles; ++cycle) 
            {
                applyErrors<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_states, d_qubits, ps[i]);                   //qubit errors
                cudaDeviceSynchronize();
                calculateSyndrome<<<(N+255)/256,256>>>(d_stabInclusionLookup, d_qubits, d_syndrome, d_edgeToFaces);    //measure stabilisers
                cudaDeviceSynchronize();
                applyErrors<<<(N+255)/256,256>>>(d_stabInclusionLookup, d_states, d_syndrome, qs[i]);                  //measurement errors
                cudaDeviceSynchronize();

                initVariableMessages<<<(N+255)/256,256>>>(d_stabInclusionLookup, d_variableMessages, llr0, llrq0);    //prepare initial distribution for BP
                cudaDeviceSynchronize();
                        
                //BP
                for (int iter=0; iter<bpIters; ++iter)  
                {
                    updateFactorMessages<<<(N+255)/256,256>>>(d_stabInclusionLookup, d_variableMessages, d_syndrome, 
                                                                d_factorMessages, d_edgeToFaces, d_faceToEdges, N);
                    cudaDeviceSynchronize();
                    updateVariableMessages<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_variableMessages, d_factorMessages, 
                                                                                d_faceToEdges, d_edgeToFaces, llr0);
                    cudaDeviceSynchronize();
                }
                calcMarginals<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_syndromeInclusionLookup, 
                                                        d_qubitMarginals, d_stabMarginals, d_factorMessages, llr0, llrq0, N);
                bpCorrection<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_syndromeInclusionLookup, d_qubits, 
                                                      d_qubitMarginals, d_syndrome, d_stabMarginals, d_faceToEdges);
                cudaDeviceSynchronize();
                //flip
                if (useFlip)
                {
                    for (int iter=0; iter<flipIters; ++iters)
                    {
                        if (iter % pfreq == 0) pflip<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_syndromeInclusionLookup, 
                                                                                d_qubits, d_syndrome, d_faceToEdges, d_states);
                        else flip<<<(N+255)/256,256>>>(d_qubitInclusionLookup, d_syndromeInclusionLookup, 
                                                                                d_qubits, d_syndrome, d_faceToEdges);
                    }
                }
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
    cudaFree(d_variableMessages);
    cudaFree(d_factorMessages);
    cudaFree(d_qubitMarginals);
    cudaFree(d_stabMarginals);
    cudaFree(d_faceToEdges);
    cudaFree(d_edgeToFaces);
    cudaFree(d_qubitInclusionLookup);
    cudaFree(d_stabInclusionLookup);
    cudaFree(d_logicalInclusionLookup);
    cudaFree(nOdd);
    return 0;
}
