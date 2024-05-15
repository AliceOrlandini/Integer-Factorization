#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "device_functions.h"						// synchronization


using namespace std;



/**
 * @brief Struct to store prime factors and their exponents
 */
struct factor_exponent {
    unsigned long long factor;
    int exponent;
};


#define MAX_PRIMES 100

// Global array statically allocated in the device memory
__device__ factor_exponent d_primes[MAX_PRIMES];
// Global variable to store the number of prime factors found
__device__ unsigned int d_primes_count = 0;


/**
 * @brief Function to check if a number is prime
 *
 * @param n number to check if prime
 * @return true if prime, false otherwise
 */
__device__ bool isPrime(unsigned long long n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 3 == 0) return false;
    for (unsigned long long i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}


/**
 * @brief Trial division function to find prime factors in a range
 *
 * @param start integer to start from
 * @param end integer to end at
 * @param num number to find prime factors of
 * @param primes vector to store prime factors
 */
__global__ void findPrimesInRange(unsigned long long *d_start, unsigned long long *d_end, unsigned long long num) {

    // (START) DEBUG
    // {
    //     lock_guard<mutex> lock(mtx);
    //     // Get the thread id
    //     thread::id this_id = this_thread::get_id();
    //     cout << "#START: Thread ID: " << this_id << " is running on core: " << GetCurrentProcessorNumber() << endl;
    // }
    // (END) DEBUG

    // Get the blockIdx.x
    unsigned long long j = blockIdx.x;


    // For avoiding problems of divergence, 
    // just execute the first thread of each warp
    // (in the considered architecture, the warp size is 32)
    if(threadIdx.x % 32 == 0){

        // Print the range of the current block
        // printf("Block ID: %d - Range: %llu - %llu\n", j, d_start[j], d_end[j]);

        // check all numbers in the range
        for (unsigned long long i = d_start[j]; i <= d_end[j]; i += 2) {

            if ((num % i) == 0) {

                // continue dividing as long as possible
                // this way we avoid adding the same factor multiple times
                int exponent = 0;
                while (num % i == 0) {
                    exponent++;
                    num /= i;
                }

                // 
                if (isPrime(i)) {
                    int index = atomicAdd(&d_primes_count, 1); // atomicAdd returns the old value
                    d_primes[index].factor = i; 
                    d_primes[index].exponent = exponent;
                }
            }
        }

    }

    


    // (START) DEBUG
    // {
    //     lock_guard<mutex> lock(mtx);
    //     // Get the thread id
    //     thread::id this_id = this_thread::get_id();
    //     cout << "#END: Thread ID: " << this_id << " is running on core: " << GetCurrentProcessorNumber() << endl;
    // }
    // (END) DEBUG

}


/**
 * @brief Main function for parallel factorization, using trial division algorithm
 *
 * @param num number to find prime factors of
 * @param numThreads number of threads to use
 * @return vector<factor_exponent> vector of prime factors
 */
vector<factor_exponent> CUDA_TrialDivision(unsigned long long num, int num_CUDA_blocks) {

    // vector to store the prime factors
    vector<factor_exponent> primes;

    // store the original number for later use
    unsigned long long old_num = num;

    // checking in advance if the number is divisible by 2
    // to avoid checking even numbers afterwards
    if (num % 2 == 0) {
        int exponent = 0;
        while (num % 2 == 0) {
            exponent++;
            num /= 2;
        }
        primes.push_back({ 2, exponent });
    }

    // now the interval to check is nearly halved
    // as checking divisibility by even numbers 
    // is not needed anymore

    unsigned long long sqrt_num = (unsigned long long) sqrt(num);

    // divide the work equally among the different CUDA Blocks
    unsigned long long range = sqrt_num / num_CUDA_blocks;

    unsigned long long start[num_CUDA_blocks];
    unsigned long long end[num_CUDA_blocks];

    // define the start and end of the range for the first Block
    start[0] = 3;
    end[0] = (range % 2 == 0) ? range + 1 : range;

    // define the start and end of the range for the other Blocks
    for (int i = 1; i < num_CUDA_blocks; i++) {
        start[i] = end[i - 1] + 2;
        if (range % 2 == 0) {
            end[i] = start[i] + range;
        }
        else {
            end[i] = start[i] + range - 1;
        }
    }
    end[num_CUDA_blocks - 1] = sqrt_num;

    // Creating the device counterparts of the start and end arrays
    unsigned long long* d_start;
    unsigned long long* d_end;

    cudaMalloc((void **)&d_start, sizeof(unsigned long long) * num_CUDA_blocks);
    cudaMalloc((void **)&d_end, sizeof(unsigned long long) * num_CUDA_blocks);

    cudaMemcpy(d_start, start, sizeof(unsigned long long) * num_CUDA_blocks, cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, end, sizeof(unsigned long long) * num_CUDA_blocks, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid(num_CUDA_blocks, 1, 1);
    dim3 threadsPerBlock(32, 1, 1);

    float elapsedTime;
    cudaEvent_t time_start, time_stop;
    
    cudaEventCreate(&time_start);
    cudaEventCreate(&time_stop);
    cudaEventRecord(time_start, 0);

    // call the kernel function to find the prime factors of the number
    findPrimesInRange<<<blocksPerGrid, threadsPerBlock>>>(d_start, d_end, num);

    cudaEventRecord(time_stop, 0);
    cudaEventSynchronize(time_stop);
    cudaEventElapsedTime(&elapsedTime, time_start, time_stop);

    cudaEventDestroy(time_start);
    cudaEventDestroy(time_stop);

    // DEBUG (START)
    cout<<"Time taken by the GPU: "<<elapsedTime<<" milliseconds."<<endl;
    // DEBUG (END)

    
    unsigned int host_primes_count;
    cudaMemcpyFromSymbol(&host_primes_count, d_primes_count, sizeof(unsigned int));

    factor_exponent* h_primes = new factor_exponent[host_primes_count];
    cudaMemcpyFromSymbol(h_primes, d_primes, sizeof(factor_exponent) * host_primes_count);

    for (int i = 0; i < host_primes_count; i++) {
        primes.push_back(h_primes[i]);
    }

    delete[] h_primes;
    cudaFree(d_start);
    cudaFree(d_end);

    // if primes is empty than the number is prime so add it to the vector
    if (primes.empty()) {
        primes.push_back({ num, 1 });
    }
    else {

        // check if all the factors have been found 
        // (otherwise a prime factor larger than the 
        // square root of the number is missing)

        // add the (possible) missing prime factor
        unsigned long long product = 1;
        for (vector<factor_exponent>::iterator it = primes.begin(); it != primes.end(); ++it) {

            // calculate the product of the prime factors by multiplying each 
            // factor by itself for the number of times stated by the exponent
            // not using pow() as it cuts off large numbers
            for (int i = 0; i < it->exponent; i++) {
                product *= it->factor;
            }

        }
        // if the product is different from the original number,
        // then the missing prime factor is the number divided by the product
        if (product != old_num) {
            primes.push_back({ old_num / product, 1 });
        }
    }

    return primes;
}


int main(int argc, char* argv[]) {

    // check if the number of arguments is correct
    if (argc != 4) {
        cout << "Please provide:\n1) Number of CUDA Blocks\n2) Number to be Factorized\n3) 0 or 1 to execute the program in BASH or USER mode." << endl;
        return 1;
    }

    // get the number of CUDA blocks from the command line argument
    int NUM_CUDA_BLOCKS = atoi(argv[1]);

    // get the number from the command line argument
    // and convert it to unsigned long long using stoull
    unsigned long long NUMBER = stoull(argv[2]);

    // get the mode (0: bash, 1: user)
    bool EXECUTION_MODE = atoi(argv[3]);

    // start measuring time (HOST point of view)
    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    // find the prime factors of the number 
    vector<factor_exponent> prime_factors = CUDA_TrialDivision(NUMBER, NUM_CUDA_BLOCKS);

    // stop measuring time
    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    // calculate the time duration
    chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    // depending on the execution mode, print some informations on screen
    if (EXECUTION_MODE) {
        cout << "Time taken: " << duration.count() << " milliseconds." << endl;

        cout << NUMBER << " = ";
        for (vector<factor_exponent>::iterator it = prime_factors.begin(); it != prime_factors.end(); ++it) {
            cout << it->factor << "^" << it->exponent;
            // print a * between prime factors except for the last one
            if (next(it) != prime_factors.end()) {
                cout << " * ";
            }
        }
        cout << endl;
    }
    else {
        cout << duration.count() << endl;
    }

    return 0;
}