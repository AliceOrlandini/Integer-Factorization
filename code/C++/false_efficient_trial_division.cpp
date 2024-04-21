#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>
#include <math.h>

using namespace std;

mutex mtx; // mutex for output synchronization


/**
 * @brief Struct to store prime factors and their exponents
 */
struct factor_exponent {
    unsigned long long factor;
    int exponent;
};

/**
 * @brief Function to check if a number is prime
 *
 * @param n number to check if prime
 * @return true if prime, false otherwise
 */
bool isPrime (unsigned long long n) {
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
void findPrimesInRange (unsigned long long start, unsigned long long end, unsigned long long num, vector<factor_exponent>& primes) {

    // check all numbers in the range
    for (unsigned long long i = start; i <= end; i += 2) {
        
        if ((num % i) == 0) {

            // continue dividing as long as possible
            // this way we avoid adding the same factor multiple times
            int exponent = 0;
            while (num % i == 0) { 
                exponent++;
                num /= i;
            }

            // lock the mutex to save the prime factor and its exponent
            if (isPrime(i)) {
                lock_guard<mutex> lock(mtx);
                primes.push_back({ i, exponent });
            }
        }
    }

    // Simulation of a long computation
    unsigned long long T = 999999999;
    unsigned long long sum = 0;
    for (unsigned long long i = 0; i < T; i++) {
        sum += i;
    }

}

/**
 * @brief Main function for parallel factorization, using trial division algorithm
 *
 * @param num number to find prime factors of
 * @param numThreads number of threads to use
 * @return vector<factor_exponent> vector of prime factors
 */
vector<factor_exponent> parallelTrialDivision (unsigned long long num, int numThreads) {

    vector<factor_exponent> primes;
    vector<thread> threads;

    // number of threads to be used for parallelization
    // const int numThreads = thread::hardware_concurrency();

    // store the original number for later use
    unsigned long long old_num = num;

    // checking in advance if the number is divisible by 2
    // to avoid checking even numbers in the loop
    if (num % 2 == 0) {
        int exponent = 0;
        while (num % 2 == 0) {
            exponent++;
            num /= 2;
        }
        // here we don't need to lock the mutex as we are in the main thread
        primes.push_back({ 2, exponent });
    }

    // now the interval to check is nearly halved
    // as checking divisibility by even numbers is not needed

    unsigned long long sqrt_num = (unsigned long long) sqrt(num);

    // divide the work equally among the threads
    unsigned long long range = sqrt_num / numThreads;

    // define the start and end of the range for the first thread
    unsigned long long start = 3;
    unsigned long long end = (range % 2 == 0) ? range + 1 : range;

    // create and start the threads
    for (int i = 0; i < (numThreads - 1); ++i) {

        threads.emplace_back(findPrimesInRange, start, end, num, ref(primes));

        // update the start and end for the next thread
        start = end + 2;

        if (range % 2 == 0) {
            end = start + range;
        } else {
            end = start + range - 1;
        }
    }
    // give to the last thread the remaining work
    end = sqrt_num;

    findPrimesInRange(start, end, num, primes);

    // wait for the other threads to finish
    for (thread& thread_ : threads) {
        thread_.join();
    }

    // if primes is empty than the number is prime so add it to the vector
    if (primes.empty()) {
        // again we don't need to lock the mutex as we are in the main thread
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
        if (product != old_num) {
            primes.push_back({ old_num / product, 1 });
        }
    }

    return primes;
}

int main(int argc, char* argv[]) {

    // check if the number of arguments is correct
    if (argc != 4) {
        cout << "Please provide:\n1) Number of Threads\n2) Number to be Factorized\n3) 0 or 1 to execute the program in BASH or USER mode." << endl;
        return 1;
    }

    // get the number of threads from the command line argument
    int NUM_THREADS = atoi(argv[1]);

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

    // depending on the execution mode, print some informations on screen
    if (EXECUTION_MODE) {
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
