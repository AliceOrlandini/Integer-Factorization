#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

using namespace std;
mutex mtx; // mutex for output synchronization
struct factor_exponent {
    int factor;
    int exponent;
};

/**
 * @brief Function to check if a number is prime (inefficiently)
 * 
 * @param n number to check if prime
 * @return true if prime, false otherwise
 */
bool isPrime(int n) {
    
    bool is_prime = true;
    
    // 0 and 1 are not prime numbers
    if (n == 0 || n == 1) {
        is_prime = false;
    }

    // check if n is divisible by any number from 2 to n/2
    // because if n is divisible by any number greater than n/2
    // then it would also be divisible by a number less than n/2
    // which we would have already checked
    for (int i = 2; i <= n/2; ++i) {
        if (n % i == 0) {
            is_prime = false;
            break;
        }
    }

    return is_prime;
}

/**
 * @brief Trial division function to find prime factors in a range
 * 
 * @param start integer to start from
 * @param end integer to end at
 * @param num number to find prime factors of
 * @param primes vector to store prime factors
 */
void findPrimesInRange(int start, int end, int num, vector<factor_exponent>& primes) {
    
    // check all numbers in the range
    for (int i = start; i <= end; ++i) {
        // if i in the range is prime and num is divisible by i
        // add it to the primes vector
        if (isPrime(i) && num % i == 0) {

            // continue dividing as long as possible
            // this way we avoid adding the same factor multiple times
            int exponent = 0;
            while (num % i == 0) {
                exponent++; 
                num /= i;
            }

            // lock the mutex
            {
                lock_guard<mutex> lock(mtx);
                primes.push_back({i, exponent});
            }
        }
    }
}

/**
 * @brief Main function for parallel factorization, using trial division
 * 
 * @param num number to find prime factors of
 * @param numThreads number of threads to use
 * @return vector<factor_exponent> vector of prime factors
 */
vector<factor_exponent> parallelTrialDivision(int num, int numThreads) {
    
    vector<factor_exponent> primes;
    vector<thread> threads;

    // number of threads to be used for parallelization
    // const int numThreads = thread::hardware_concurrency();

    // divide the work equally among the threads
    int range = num / numThreads;

    // define the start and end of the range for the first thread
    int start = 2;
    int end = range;

    // create and start the threads
    for (int i = 0; i < (numThreads - 1); ++i) {
        threads.emplace_back(findPrimesInRange, start, end, num, ref(primes));
        
        // Print thread identifier
        // cout <<"Thread["<<i<<"] ID: " << threads[i].get_id() << endl;

        // update the start and end for the next thread
        start = end + 1;
        end = start + range - 1;
    }
    end = num;

    // Last thread is the main thread
    findPrimesInRange(start, end, num, primes);

    // wait for the other threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    return primes;
}

int main(int argc, char *argv[]) {
    
    if (argc != 4) {
        cout << "Please provide: Number of Threads, Number to be Factorized and 0 or 1 to execute the program in BASH or USER mode." << endl;
        return 1;
    }

    // get the number of threads from the command line argument
    int NUM_THREADS = atoi(argv[1]);

    // get the number from the command line argument
    int NUMBER = atoi(argv[2]);

    // get the mode (0: bash, 1: user)
    bool EXECUTION_MODE = atoi(argv[3]);

    // start measuring time
    auto start = chrono::high_resolution_clock::now();
    // chrono::steady_clock::time_point start = chrono::high_resolution_clock::now();
    
    // find the prime factors of the number
    vector<factor_exponent> factors = parallelTrialDivision(NUMBER, NUM_THREADS);

    // stop measuring time
    auto end = chrono::high_resolution_clock::now();
    // chrono::steady_clock::time_point end = chrono::high_resolution_clock::now();
    

    // calculate the time duration
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    // chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    if(EXECUTION_MODE) {
        cout << "Time taken: " << duration.count() << " milliseconds." << endl;

        cout << NUMBER << " = ";
        for (auto it = factors.begin(); it != factors.end(); ++it) {
            cout << it->factor << "^" << it->exponent;
            // print a * between factors except for the last one
            if (next(it) != factors.end()) {
                cout << " * ";
            }
        }
        cout << endl;
    } else {
        cout << duration.count() << endl;
    }
    
    return 0;
}
