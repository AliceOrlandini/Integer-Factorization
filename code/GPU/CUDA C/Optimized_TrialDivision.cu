/**
 * [CUDA OPTIMIZED VERSION]
 * Optimized CUDA version of the parallel trial division algorithm.
 * 
 */

#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <math.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_runtime_api.h"
#include "device_functions.h"						// CUDA device functions for synchronization


using namespace std;

/**
 * @brief Struct to store prime factors and their exponents
 */
struct factor_exponent {
    unsigned long long factor; // Prime factor
    int exponent; // Exponent of the prime factor
};

// Number of threads per block 
// It is better to use a multiple of 32
// as the number of threads per warp is 32
#define NUM_THREADS_PER_BLOCK 64

// Assumption: maximum number of distinct
// primes that factorize the input number
#define MAX_PRIMES 30


// Global array statically allocated in the device memory to store prime factors
__device__ factor_exponent d_primes[MAX_PRIMES];
// Global variable to store the number of prime factors found
__device__ unsigned int d_primes_count = 0;


/**
 * @brief Device function to check if a number is prime
 * 
 * @param n Number to check for primality
 * @return true if the number is prime, false otherwise
 */
__device__ bool isPrime(unsigned long long n) {
    // Numbers less than or equal to 1 are not prime
    if (n <= 1) return false;
    // Numbers 2 and 3 are prime
    if (n <= 3) return true;
    // Multiples of 3 are not prime
    if (n % 3 == 0) return false;
    // Check for factors from 5 to sqrt(n)
    for (unsigned long long i = 5; i * i <= n; i += 6) {
        // If divisible by i or (i + 2), n is not prime
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}


/**
 * @brief Kernel function to find prime factors of a number in a given range
 *
 * @param d_start array of start values for each thread
 * @param d_end array of end values for each thread
 * @param num number to find prime factors of
 */
__global__ void findPrimesInRange(unsigned long long *d_start, unsigned long long *d_end, unsigned long long num) {

    // Get grid index
    unsigned long long j = threadIdx.x + blockIdx.x * blockDim.x;

    // (START) DEBUG
    // Print the range of the current Thread
    // printf("Grid Index: %llu - Range: %llu - %llu\n", j, d_start[j], d_end[j]);
    // (END) DEBUG

    // Check all numbers in the range
    for (unsigned long long i = d_start[j]; i <= d_end[j]; i += 2) {

        if ((num % i) == 0) {

            // Divergence of threads inside the same warp
            // happens here. However, it is not frequent
            // if compared to the huge number of times
            // that num does not divide i.


            // Continue dividing as long as possible
            // this way we avoid adding the same factor multiple times
            int exponent = 0;
            while (num % i == 0) {
                exponent++;
                num /= i;
            }

            if (isPrime(i)) {
                // Accessing the global memory
                int index = atomicAdd(&d_primes_count, 1); // atomicAdd returns the old value
                d_primes[index].factor = i; 
                d_primes[index].exponent = exponent;
            }

        }

    }

}


/**
 * @brief Parallel factorization using trial division
 *
 * @param num Number to be factorized
 * @param num_CUDA_blocks Number of CUDA blocks to use
 * @param EXECUTION_MODE Execution mode (0: BASH, 1: USER)
 * @return vector<factor_exponent> vector of prime factors
 */
vector<factor_exponent> CUDA_TrialDivision(unsigned long long num, int num_CUDA_blocks, bool EXECUTION_MODE) {

    // Vector to store the prime factors
    vector<factor_exponent> primes;

    // Store the original number for later use
    unsigned long long old_num = num;

    // Checking in advance if the number is divisible by 2
    // to avoid checking even numbers afterwards
    if (num % 2 == 0) {
        int exponent = 0;
        while (num % 2 == 0) {
            exponent++;
            num /= 2;
        }
        primes.push_back({ 2, exponent });
    }

    // Now the interval to check is nearly halved
    // as checking divisibility by even numbers 
    // is not needed anymore

    unsigned long long sqrt_num = (unsigned long long) sqrt(num);

    unsigned long long num_CUDA_threads = num_CUDA_blocks * NUM_THREADS_PER_BLOCK;

    // divide the work equally among the different CUDA Threads
    unsigned long long range = sqrt_num / num_CUDA_threads;

    unsigned long long start[num_CUDA_threads];
    unsigned long long end[num_CUDA_threads];

    // Define the start and end of the range for the first CUDA Thread
    start[0] = 3;
    end[0] = (range % 2 == 0) ? range + 1 : range;

    // Define the start and end of the range for the other CUDA Threads
    for (int i = 1; i < num_CUDA_threads; i++) {
        start[i] = end[i - 1] + 2;
        if (range % 2 == 0) {
            end[i] = start[i] + range;
        }
        else {
            end[i] = start[i] + range - 1;
        }
    }
    end[num_CUDA_threads - 1] = sqrt_num;

    // Creating the device counterparts of the start and end arrays
    unsigned long long* d_start;
    unsigned long long* d_end;

    // Allocate memory for the device arrays
    cudaMalloc((void **)&d_start, sizeof(unsigned long long) * num_CUDA_threads);
    cudaMalloc((void **)&d_end, sizeof(unsigned long long) * num_CUDA_threads);

    // Copy the start and end arrays to the device
    cudaMemcpy(d_start, start, sizeof(unsigned long long) * num_CUDA_threads, cudaMemcpyHostToDevice);
    cudaMemcpy(d_end, end, sizeof(unsigned long long) * num_CUDA_threads, cudaMemcpyHostToDevice);

    // Dimensions of the CUDA grid
    dim3 blocksPerGrid(num_CUDA_blocks, 1, 1);
    // Dimensions of the CUDA block
    dim3 threadsPerBlock(NUM_THREADS_PER_BLOCK, 1, 1);

    // Measure the time taken by the GPU
    float elapsedTime;
    cudaEvent_t time_start, time_stop;
    
    // Create the events
    cudaEventCreate(&time_start);
    cudaEventCreate(&time_stop);
    // Record the start event
    cudaEventRecord(time_start, 0);

    // Call the kernel function to find the prime factors of the number
    findPrimesInRange<<<blocksPerGrid, threadsPerBlock>>>(d_start, d_end, num);

    // Record the stop event
    cudaEventRecord(time_stop, 0);
    // Synchronize the stop event
    cudaEventSynchronize(time_stop);
    // Calculate the elapsed time
    cudaEventElapsedTime(&elapsedTime, time_start, time_stop);

    // Destroy the events
    cudaEventDestroy(time_start);
    cudaEventDestroy(time_stop);

    if (EXECUTION_MODE)
        cout<<"Time taken by the GPU: "<<elapsedTime<<" milliseconds."<<endl;
    else
        cout<<elapsedTime<<",";

    
    unsigned int host_primes_count;
    // Copy the number of prime factors found to the host
    cudaMemcpyFromSymbol(&host_primes_count, d_primes_count, sizeof(unsigned int));

    // Copy the prime factors found to the host
    factor_exponent* h_primes = new factor_exponent[host_primes_count];
    cudaMemcpyFromSymbol(h_primes, d_primes, sizeof(factor_exponent) * host_primes_count);

    for (int i = 0; i < host_primes_count; i++) {
        primes.push_back(h_primes[i]);
    }

    delete[] h_primes;
    // Free the device memory
    cudaFree(d_start);
    cudaFree(d_end);

    // If no prime factors found, num itself is prime
    if (primes.empty()) {
        // Again we don't need to lock the mutex as we are in the main thread
        primes.push_back({ num, 1 });
    } 
    else {
        // Check if all the prime factors have been found 
        // (otherwise a prime factor larger than the 
        // square root of the number is missing)

        unsigned long long product = 1;
        // Calculate the product of found prime factors
        for (vector<factor_exponent>::iterator it = primes.begin(); it != primes.end(); ++it) {
            // Calculate the product of the prime factors by multiplying each 
            // factor by itself for the number of times stated by the exponent
            // Not using pow() as it cuts off large numbers
            for (int i = 0; i < it->exponent; i++) {
                product *= it->factor;
            }
        }
        // If product is different from the original number, add the missing prime factor
        if (product != old_num) {
            primes.push_back({ old_num / product, 1 });
        }
    }

    // Return the vector of prime factors and their exponents
    return primes;
}


int main(int argc, char* argv[]) {

    // Check for correct number of arguments
    if (argc != 4) {
        cout << "Please provide:\n1) Number of CUDA Blocks\n2) Number to be Factorized\n3) 0 or 1 to execute the program in BASH or USER mode." << endl;
        // Exit if arguments are incorrect
        return 1;
    }

    // Get the number of CUDA blocks from the command line argument
    int NUM_CUDA_BLOCKS = atoi(argv[1]);

    // Get the number to be factorized from command line arguments
    // And convert it to unsigned long long using stoull
    unsigned long long NUMBER = stoull(argv[2]);

    // Get the execution mode (0:BASH or 1:USER) from command line arguments
    bool EXECUTION_MODE = atoi(argv[3]);

    // Start measuring time (HOST point of view)
    chrono::steady_clock::time_point start = chrono::steady_clock::now();

    // Factorize the number using the GPU
    vector<factor_exponent> prime_factors = CUDA_TrialDivision(NUMBER, NUM_CUDA_BLOCKS, EXECUTION_MODE);

    // Stop measuring time
    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    // Calculate the time duration
    chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    // If USER mode
    if (EXECUTION_MODE) {
        cout << "Time taken: " << duration.count() << " milliseconds." << endl;

        cout << NUMBER << " = ";
        for (vector<factor_exponent>::iterator it = prime_factors.begin(); it != prime_factors.end(); ++it) {
            // Print each prime factor and its exponent
            cout << it->factor << "^" << it->exponent;
            if (next(it) != prime_factors.end()) {
                // Print * between factors
                cout << " * ";
            }
        }
        cout << endl;
    }
    else {
        // If BASH mode, print only the time taken
        cout << duration.count() << endl;
    }

    return 0;
}