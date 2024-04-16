#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <math.h>

using namespace std;


mutex mtx; // mutex for output synchronization

struct factor_exponent {
    unsigned long long factor;
    int exponent;
};

/**
 * @brief Function to check if a number is prime (inefficiently)
 * 
 * @param n number to check if prime
 * @return true if prime, false otherwise
 */
bool isPrime(unsigned long long n) {
    
    bool is_prime = true;
    
    // 0 and 1 are not prime numbers
    if (n == 0 || n == 1) {
        is_prime = false;
    }

    for (unsigned long long i = 2; i * i <= n; ++i) {
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
void findPrimesInRange(unsigned long long start, unsigned long long end, unsigned long long num, vector<factor_exponent>& primes) {
    
    // check all numbers in the range
    for (unsigned long long i = start; i <= end; ++i) {
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
vector<factor_exponent> parallelTrialDivision(unsigned long long num, int numThreads) {
    
    vector<factor_exponent> primes;
    vector<thread> threads;

    unsigned long long sqrt_num = (unsigned long long) sqrt(num);

    // divide the work equally among the threads
    unsigned long long range = sqrt_num / numThreads;

    // define the start and end of the range for the first thread
    unsigned long long start = 2;
    unsigned long long end = range;

    // create and start the threads
    for (int i = 0; i < (numThreads - 1); ++i) {
        threads.emplace_back(findPrimesInRange, start, end, num, ref(primes));
        
        // update the start and end for the next thread
        start = end + 1;
        end = start + range - 1;
    }
    // give to the last thread the remaining work
    end = sqrt_num;

    // Last thread is the main thread
    findPrimesInRange(start, end, num, primes);

    // wait for the other threads to finish
    for (thread& thread_ : threads) {
        thread_.join();
    }

    // if primes is empty than the number is prime so add it to the vector
    if (primes.empty()) {
        // We do not need to lock the mutext because at 
        //this point all the threads have finished
        primes.push_back({ num, 1 });
    } else {

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
        if (product != num) {
            primes.push_back({ num / product, 1 });
        }
    }

    return primes;
}

int main(int argc, char *argv[]) {
    
    if (argc != 4) {
        cout << "Please provide: Number of Threads, Number to be Factorized and 0 or 1 to execute the program in BASH or USER mode." << endl;
        return 1;
    }

    // get the number of threads from the command line argument
    unsigned long long NUM_THREADS = atoi(argv[1]);

    // get the number from the command line argument
    // and convert it to unsigned long long using stoull
    unsigned long long NUMBER = stoull(argv[2]);

    // get the mode (0: bash, 1: user)
    bool EXECUTION_MODE = atoi(argv[3]);
    
    // start measuring time
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    
    // find the prime factors of the number
    vector<factor_exponent> factors = parallelTrialDivision(NUMBER, NUM_THREADS);

    // stop measuring time
    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    // calculate the time duration
    chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    if(EXECUTION_MODE) {
        cout << "Time taken: " << duration.count() << " milliseconds." << endl;

        cout << NUMBER << " = ";
        for (vector<factor_exponent>::iterator it = factors.begin(); it != factors.end(); ++it) {
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
