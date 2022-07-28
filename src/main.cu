#include<random>
#include<iostream>
#include "code.h"
#include "decode.cuh"

int main(int argc, char *argv[])
{
    if (argc != 9)
    {
        std::cout << "Invalid number of arguments." << '\n';
        return 1;
    }

    double pLower = std::atof(argv[1]);  //lower value for error probability p
    double pUpper = std::atof(argv[2]);  //upper value for p
    int nps = std::atoi(argv[3]);        //number of values for p in range pLower <= p <= pUpper
    double alpha = std::atof(argv[4]);   //measurement error probability q = alpha*p
    int runs = std::atoi(argv[5]);       //number of repeats of simulation
    int cycles = std::atoi(argv[6]);     //code cycles per simulation
    int apps = std::atoi(argv[7]);       //applications of decoding rule per code cycle
    int pfreq = std::atoi(argv[8]);      //apply pFlip instead of flip every pfreq applications

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
    Code code("parity_check_matrices/lifted_product_[[416,18,20]]");

    //for copying out later
    int qubitsX[code.N] = {};
    int qubitsZ[code.N] = {};
    int syndromeX[code.M_X] = {};
    int syndromeZ[code.M_Z] = {};
    //pointers for arrays on device
    int *d_qubitsX, *d_qubitsZ, *d_syndromeX, *d_syndromeZ;
    int *d_bitToXChecks, *d_bitToZChecks, *d_xCheckToBits, *d_zCheckToBits;

    //don't need to copy for these, just set to all zeros on device (later)
    cudaMalloc(&d_qubitsX, code.N*sizeof(int));
    cudaMalloc(&d_qubitsZ, code.N*sizeof(int));
    cudaMalloc(&d_syndromeX, code.M_X*sizeof(int));
    cudaMalloc(&d_syndromeZ, code.M_Z*sizeof(int));

    //these get copied to device from initialised versions in code object
    cudaMalloc(&d_bitToXChecks, code.maxBitDegreeX*code.N*sizeof(int));
    cudaMemcpy(d_bitToXChecks, code.bitToXChecks[0], 
                code.maxBitDegreeX*code.N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_bitToZChecks, code.maxBitDegreeZ*code.N*sizeof(int));
    cudaMemcpy(d_bitToZChecks, code.bitToZChecks[0], 
                code.maxBitDegreeZ*code.N*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_xCheckToBits, code.maxCheckDegreeX*code.M_X*sizeof(int));
    cudaMemcpy(d_xCheckToBits, code.xCheckToBits[0], 
                code.maxCheckDegreeX*code.M_X*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_zCheckToBits, code.maxCheckDegreeZ*code.M_Z*sizeof(int));
    cudaMemcpy(d_zCheckToBits, code.zCheckToBits[0],
                code.maxCheckDegreeZ*code.M_Z*sizeof(int), cudaMemcpyHostToDevice);

    //setup state array for device-side random number generation
    std::random_device rd{};
    curandState_t *d_states;
    cudaMalloc(&d_states, code.N*sizeof(curandState_t));
    createStates<<<(code.N+255)/256,256>>>(code.N, rd(), d_states);
    cudaDeviceSynchronize();

    for (int i=0; i<nps; ++i)
    {
        for (int run=0; run<runs; ++run)
        {
            //set qubits and syndrome to all zeros 
            wipeArray<<<(code.N+255)/256,256>>>(code.N, d_qubitsX);
            wipeArray<<<(code.N+255)/256,266>>>(code.N, d_qubitsZ);
            cudaDeviceSynchronize();
        
            for (int cycle=1; cycle<cycles+1; ++cycle) //starts on 1 instead of 0 so cycle % efreq != 0 on the first loop
            {
                depolErrors<<<(code.N+255)/256,256>>>(code.N, d_states, d_qubitsX, d_qubitsZ, ps[i]);                                 //qubit errors
                cudaDeviceSynchronize();
                calculateSyndrome<<<(code.N+255)/256,256>>>(code.M_Z, d_qubitsX, d_syndromeZ, d_zCheckToBits, code.maxCheckDegreeZ);  //measure stabilisers
                calculateSyndrome<<<(code.N+255)/256,256>>>(code.M_X, d_qubitsZ, d_syndromeX, d_xCheckToBits, code.maxCheckDegreeX);
                cudaDeviceSynchronize();
                arrayErrors<<<(code.N+255)/256,256>>>(code.M_Z, d_states, d_syndromeZ, qs[i]);                                        //measurement errors
                arrayErrors<<<(code.N+255)/256,256>>>(code.M_X, d_states, d_syndromeX, qs[i]);
                cudaDeviceSynchronize();
                        
                for (int app=1; app<apps+1; ++app)  //start on 1 instead of 0 so app % pfreq != 0 on the first loop (unless pfreq=1)
                {
                    //use pflip every pfreq applications, otherwise use regular flip
                    if (app % pfreq == 0)
                    {
                        pflip<<<(code.N+255)/256,256>>>(code.N, code.M_Z, d_states, d_qubitsX, d_syndromeZ, d_bitToZChecks, code.maxBitDegreeZ);
                        pflip<<<(code.N+255)/256,256>>>(code.N, code.M_X, d_states, d_qubitsZ, d_syndromeX, d_bitToXChecks, code.maxBitDegreeX);
                    }
                    else
                    {
                        flip<<<(code.N+255)/256,256>>>(code.N, code.M_Z, d_qubitsX, d_syndromeZ, d_bitToZChecks, code.maxBitDegreeZ);
                        flip<<<(code.N+255)/256,256>>>(code.N, code.M_X, d_qubitsZ, d_syndromeX, d_bitToXChecks, code.maxBitDegreeX);
                    }
                    cudaDeviceSynchronize();
                }
            }

            cudaMemcpy(qubitsX, d_qubitsX, code.N*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(syndromeZ, d_syndromeZ, code.M_Z*sizeof(int), cudaMemcpyDeviceToHost);
            std::cout << ps[i] << ",q,X";
            for (int j=0; j<code.N; ++j) std::cout << ',' << qubitsX[j];
            std::cout << '\n';
            std::cout << ps[i] << ",s,Z";
            for (int j=0; j<code.M_Z; ++j) std::cout << ',' << syndromeZ[j];
            std::cout << '\n';
            cudaMemcpy(qubitsZ, d_qubitsZ, code.N*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(syndromeX, d_syndromeX, code.M_X*sizeof(int), cudaMemcpyDeviceToHost);
            std::cout << ps[i] << ",q,Z";
            for (int j=0; j<code.N; ++j) std::cout << ',' << qubitsZ[j];
            std::cout << '\n';
            std::cout << ps[i] << ",s,X";
            for (int j=0; j<code.M_X; ++j) std::cout << ',' << syndromeX[j];
            std::cout << '\n';
        }
    }

    cudaFree(d_qubitsX);
    cudaFree(d_qubitsZ);
    cudaFree(d_syndromeX);
    cudaFree(d_syndromeZ);
    cudaFree(d_bitToXChecks);
    cudaFree(d_bitToZChecks);
    cudaFree(d_xCheckToBits);
    cudaFree(d_zCheckToBits);
    return 0;
}
