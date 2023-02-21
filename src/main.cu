#include<random>
#include<iostream>
#include "code.h"
#include "decode.cuh"

int main(int argc, char *argv[])
{
    if (argc != 13)
    {
        std::cout << "Invalid number of arguments." << '\n';
        return 1;
    }

    double pLower = std::atof(argv[1]);  //lower value for error probability p
    double pUpper = std::atof(argv[2]);  //upper value for p
    int nps = std::atoi(argv[3]);        //number of values for p in range pLower <= p <= pUpper
    double beta = std::atof(argv[4]);    //measurement error probability q = beta*p
    int runs = std::atoi(argv[5]);       //number of repeats of simulation
    int cycles = std::atoi(argv[6]);     //code cycles per simulation
    int useBP = std::atoi(argv[7]);      //use BP in decoding? (0 = no, 1 = tanh BP, 2 = min sum BP)
    int bpIters = std::atoi(argv[8]);    //BP iterations per code cycle
    int alpha = std::atof(argv[9]);      //alpha parameter for minsum BP
    int useFlip = std::atoi(argv[10]);    //use flip in decoding? (0 = np, 1 = yes)
    int flipIters = std::atoi(argv[11]); //flip iterations per code cycle
    int pfreq = std::atoi(argv[12]);     //apply p-flip instead of flip every pfreq applications

    double pRange = pUpper - pLower;
    double pStep;
    if (nps == 1) pStep = 0;
    else pStep = pRange/(nps-1); 
    double ps[nps];
    double qs[nps];
    for (int i=0; i<nps; ++i)
    {
        ps[i] = pLower + i*pStep;
        qs[i] = beta*ps[i];
    }
      
    //build code info 
    Code code("parity_check_matrices/lifted_product_[[416,18,20]]",416);

    //for copying out later
    int variablesX[code.N_X] = {};     
    int variablesZ[code.N_Z] = {};
    int factorsX[code.M_X] = {};
    int factorsZ[code.M_Z] = {};
    //pointers for arrays on device
    int *d_variablesX, *d_variablesZ, *d_factorsX, *d_factorsZ;
    int *d_variableDegreesX, *d_variableToFactorsX, *d_variableDegreesZ, *d_variableToFactorsZ;
    int *d_factorDegreesX, *d_factorToVariablesX, *d_factorDegreesZ, *d_factorToVariablesZ;
    int *d_variableToPosX, *d_variableToPosZ, *d_factorToPosX, *d_factorToPosZ;
    //BP message array pointers
    double *d_variableMessagesX, *d_variableMessagesZ, *d_factorMessagesX, *d_factorMessagesZ; 
    double *d_marginalsX, *d_marginalsZ;

    //don't need to copy for these, just set to all zeros on device (later)
    cudaMalloc(&d_variablesX, code.N_X*sizeof(int));
    cudaMalloc(&d_variablesZ, code.N_Z*sizeof(int));
    cudaMalloc(&d_factorsX, code.M_X*sizeof(int));
    cudaMalloc(&d_factorsZ, code.M_Z*sizeof(int));

    //these get copied to device from initialised versions in code object
    cudaMalloc(&d_variableDegreesX, code.N_X*sizeof(int));
    cudaMemcpy(d_variableDegreesX, code.variableDegreesX,
                code.N_X*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_variableToFactorsX, code.maxVariableDegreeX*code.N_X*sizeof(int));
    cudaMemcpy(d_variableToFactorsX, code.variableToFactorsX[0], 
                code.maxVariableDegreeX*code.N_X*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_variableDegreesZ, code.N_Z*sizeof(int));
    cudaMemcpy(d_variableDegreesZ, code.variableDegreesZ,
                code.N_Z*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_variableToFactorsZ, code.maxVariableDegreeZ*code.N_Z*sizeof(int));
    cudaMemcpy(d_variableToFactorsZ, code.variableToFactorsZ[0], 
                code.maxVariableDegreeZ*code.N_Z*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_factorDegreesX, code.M_X*sizeof(int));
    cudaMemcpy(d_factorDegreesX, code.factorDegreesX,
                code.M_X*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_factorToVariablesX, code.maxFactorDegreeX*code.M_X*sizeof(int));
    cudaMemcpy(d_factorToVariablesX, code.factorToVariablesX[0], 
                code.maxFactorDegreeX*code.M_X*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_factorDegreesZ, code.M_Z*sizeof(int));
    cudaMemcpy(d_factorDegreesZ, code.factorDegreesZ,
                code.M_Z*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_factorToVariablesZ, code.maxFactorDegreeZ*code.M_Z*sizeof(int));
    cudaMemcpy(d_factorToVariablesZ, code.factorToVariablesZ[0], 
                code.maxFactorDegreeZ*code.M_Z*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_variableToPosX, code.maxVariableDegreeX*code.N_X*sizeof(int));
    cudaMemcpy(d_variableToPosX, code.variableToPosX[0],
                code.maxVariableDegreeX*code.N_X*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_variableToPosZ, code.maxVariableDegreeZ*code.N_Z*sizeof(int));
    cudaMemcpy(d_variableToPosZ, code.variableToPosZ[0],
                code.maxVariableDegreeZ*code.N_Z*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_factorToPosX, code.maxFactorDegreeX*code.M_X*sizeof(int));
    cudaMemcpy(d_factorToPosX, code.factorToPosX[0],
                code.maxFactorDegreeX*code.M_X*sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc(&d_factorToPosZ, code.maxFactorDegreeZ*code.M_Z*sizeof(int));
    cudaMemcpy(d_factorToPosZ, code.factorToPosZ[0],
            code.maxFactorDegreeZ*code.M_Z*sizeof(int), cudaMemcpyHostToDevice);

    //These also get initialised on device
    cudaMalloc(&d_variableMessagesX, code.maxFactorDegreeZ*code.M_Z*sizeof(double));
    cudaMalloc(&d_variableMessagesZ, code.maxFactorDegreeX*code.M_X*sizeof(double));
    cudaMalloc(&d_factorMessagesX, code.maxVariableDegreeZ*code.N_Z*sizeof(double));
    cudaMalloc(&d_factorMessagesZ, code.maxVariableDegreeX*code.N_X*sizeof(double));
    cudaMalloc(&d_marginalsX, code.N_X*sizeof(double));
    cudaMalloc(&d_marginalsZ, code.N_Z*sizeof(double));

    //setup state array for device-side random number generation
    std::random_device rd{};
    curandState_t *d_states;
    //just need at least as many states as the largest array
    if (code.N_X > code.N_Z) 
    {
        cudaMalloc(&d_states, code.N_X*sizeof(curandState_t));
        createStates<<<(code.N_X+255)/256,256>>>(code.N_X, rd(), d_states);
    }
    else 
    {
        cudaMalloc(&d_states, code.N_Z*sizeof(curandState_t));
        createStates<<<(code.N_Z+255)/256,256>>>(code.N_Z, rd(), d_states);
    }
    cudaDeviceSynchronize();

    for (int i=0; i<nps; ++i)
    {
        double llrp0 = log10((1-ps[i])/ps[i]);
        double llrq0 = log10((1-qs[i])/qs[i]);
        for (int run=0; run<runs; ++run)
        {
            //set qubits to all zeros 
            wipeArray<<<(code.N_X+255)/256,256>>>(code.N_X, d_variablesX);
            wipeArray<<<(code.N_Z+255)/256,266>>>(code.N_Z, d_variablesZ);
            cudaDeviceSynchronize();
        
            for (int cycle=0; cycle<cycles; ++cycle) 
            {
                depolErrors<<<(code.nQubits+255)/256,256>>>(code.nQubits, d_states, d_variablesX, d_variablesZ, ps[i]);                 //qubit errors
                measErrors<<<(code.N_X+255)/256,256>>>(code.nQubits, code.nChecksZ, d_states, d_variablesX, qs[i]);                     //Z measurement errors
                measErrors<<<(code.N_Z+255)/256,256>>>(code.nQubits, code.nChecksX, d_states, d_variablesZ, qs[i]);                     //X measurement errors
                cudaDeviceSynchronize();
                calculateSyndrome<<<(code.M_Z+255)/256,256>>>(code.M_Z, d_variablesX, d_factorsZ,                                       //calculate checks
                                        d_factorToVariablesZ, d_factorDegreesZ, code.maxFactorDegreeZ);                                 //(inc. metachecks)
                calculateSyndrome<<<(code.M_X+255)/256,256>>>(code.M_X, d_variablesZ, d_factorsX, 
                                        d_factorToVariablesX, d_factorDegreesX, code.maxFactorDegreeX);
                cudaDeviceSynchronize();

                //prepare initial distributions for BP
                initVariableMessages<<<(code.M_Z+255)/256,256>>>(code.M_Z, code.nChecksZ, d_variableMessagesX, d_factorDegreesZ,
                        code.maxFactorDegreeZ, llrp0, llrq0);
                initVariableMessages<<<(code.M_X+255)/256,256>>>(code.M_X, code.nChecksX, d_variableMessagesZ, d_factorDegreesX,
                        code.maxFactorDegreeX, llrp0, llrq0);
                cudaDeviceSynchronize();

                //BP
                if (useBP)
                {
                    for (int iter=0; iter<bpIters; ++iter)
                    {
                        if (useBP == 1)
                        {
                            updateFactorMessagesTanh<<<(code.M_Z+255)/256,256>>>(code.M_Z, d_variableMessagesX, d_factorMessagesZ, d_factorsZ,
                                    d_factorToVariablesZ, d_factorDegreesZ, code.maxFactorDegreeZ, d_factorToPosZ, code.maxVariableDegreeX);
                            updateFactorMessagesTanh<<<(code.M_X+255)/256,256>>>(code.M_X, d_variableMessagesZ, d_factorMessagesX, d_factorsX,
                                    d_factorToVariablesX, d_factorDegreesX, code.maxFactorDegreeX, d_factorToPosX, code.maxVariableDegreeZ);
                        }
                        else if (useBP == 2)
                        {
                            updateFactorMessagesMinSum<<<(code.M_Z+255)/256,256>>>(alpha, code.M_Z, d_variableMessagesX, d_factorMessagesZ, d_factorsZ,
                                    d_factorToVariablesZ, d_factorDegreesZ, code.maxFactorDegreeZ, d_factorToPosZ, code.maxVariableDegreeX);
                            updateFactorMessagesMinSum<<<(code.M_X+255)/256,256>>>(alpha, code.M_X, d_variableMessagesZ, d_factorMessagesX, d_factorsX,
                                    d_factorToVariablesX, d_factorDegreesX, code.maxFactorDegreeX, d_factorToPosX, code.maxVariableDegreeZ);
                        }
                        cudaDeviceSynchronize();
                        updateVariableMessages<<<(code.N_X+255)/256,256>>>(code.N_X, code.nQubits, d_factorMessagesZ, d_variableMessagesX, 
                                d_variableToFactorsX, d_variableDegreesX, code.maxVariableDegreeX, d_variableToPosX, code.maxFactorDegreeZ, llrp0, llrq0);
                        updateVariableMessages<<<(code.N_Z+255)/256,256>>>(code.N_Z, code.nQubits, d_factorMessagesX, d_variableMessagesZ,
                                d_variableToFactorsZ, d_variableDegreesZ, code.maxVariableDegreeZ, d_variableToPosZ, code.maxFactorDegreeX, llrp0, llrq0);
                        cudaDeviceSynchronize();
                    }
                    calcMarginals<<<(code.N_X+255)/256,256>>>(code.N_X, code.nQubits, d_marginalsX, d_factorMessagesZ, 
                                                                d_variableDegreesX, code.maxVariableDegreeX, llrp0, llrq0);
                    calcMarginals<<<(code.N_Z+255)/256,256>>>(code.N_Z, code.nQubits, d_marginalsZ, d_factorMessagesX, 
                                                                d_variableDegreesZ, code.maxVariableDegreeZ, llrp0, llrq0);
                    cudaDeviceSynchronize();    
                    bpCorrection<<<(code.N_X+255)/256,256>>>(code.nQubits, code.nChecksZ, d_marginalsX, d_variablesX, d_factorsZ,
                            d_variableToFactorsX, d_variableDegreesX, code.maxVariableDegreeX);
                    bpCorrection<<<(code.N_Z+255)/256,256>>>(code.nQubits, code.nChecksX, d_marginalsZ, d_variablesZ, d_factorsX,
                            d_variableToFactorsZ, d_variableDegreesZ, code.maxVariableDegreeZ);
                    cudaDeviceSynchronize();
                }
                //flip
                if (useFlip)
                {
                    for (int iter=0; iter<flipIters; ++iter)
                    {
                        //if we used BP we can run pflip straight away, otherwise do some normal flip first
                        if ((useBP == 0 && (iter+1) % pfreq == 0) || (useBP == 1 && iter % pfreq == 0))
                        {
                            pflip<<<(code.N_X+255)/256,256>>>(code.nQubits, d_states, d_variablesX, d_factorsZ,
                                        d_variableToFactorsX, d_variableDegreesX, code.maxVariableDegreeX);
                            pflip<<<(code.N_Z+255)/256,256>>>(code.nQubits, d_states, d_variablesZ, d_factorsX,
                                        d_variableToFactorsZ, d_variableDegreesZ, code.maxVariableDegreeZ);
                        }
                        else
                        {
                            flip<<<(code.N_X+255)/256,256>>>(code.nQubits, d_variablesX, d_factorsZ, 
                                    d_variableToFactorsX, d_variableDegreesX, code.maxVariableDegreeX);
                            flip<<<(code.N_Z+255)/256,256>>>(code.nQubits, d_variablesZ, d_factorsX,
                                    d_variableToFactorsZ, d_variableDegreesZ, code.maxVariableDegreeZ);
                        }
                        cudaDeviceSynchronize();
                    }
                }
            }

            cudaMemcpy(variablesX, d_variablesX, code.N_X*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(factorsZ, d_factorsZ, code.M_Z*sizeof(int), cudaMemcpyDeviceToHost);
            std::cout << ps[i] << ',' << run << ",q,X";
            for (int j=0; j<code.nQubits; ++j) std::cout << ',' << variablesX[j];
            std::cout << '\n';
            std::cout << ps[i] << ',' << run << ",s,Z";
            for (int j=0; j<code.nChecksZ; ++j) std::cout << ',' << factorsZ[j];
            std::cout << '\n';
            cudaMemcpy(variablesZ, d_variablesZ, code.N_Z*sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(factorsX, d_factorsX, code.M_X*sizeof(int), cudaMemcpyDeviceToHost);
            std::cout << ps[i] << ',' << run << ",q,Z";
            for (int j=0; j<code.nQubits; ++j) std::cout << ',' << variablesZ[j];
            std::cout << '\n';
            std::cout << ps[i] << ',' << run << ",s,X";
            for (int j=0; j<code.nChecksX; ++j) std::cout << ',' << factorsX[j];
            std::cout << '\n';
        }
    }

    cudaFree(d_variablesX);
    cudaFree(d_variablesZ);
    cudaFree(d_factorsX);
    cudaFree(d_factorsZ);
    cudaFree(d_variableDegreesX);
    cudaFree(d_variableToFactorsX);
    cudaFree(d_variableDegreesZ);
    cudaFree(d_variableToFactorsZ);
    cudaFree(d_factorDegreesX);
    cudaFree(d_factorToVariablesX);
    cudaFree(d_factorDegreesZ);
    cudaFree(d_factorToVariablesZ);
    cudaFree(d_variableToPosX);
    cudaFree(d_variableToPosZ);
    cudaFree(d_factorToPosX);
    cudaFree(d_factorToPosZ);
    cudaFree(d_variableMessagesX);
    cudaFree(d_variableMessagesZ);
    cudaFree(d_factorMessagesX);
    cudaFree(d_factorMessagesZ);
    cudaFree(d_marginalsX);
    cudaFree(d_marginalsZ);

    return 0;
}
