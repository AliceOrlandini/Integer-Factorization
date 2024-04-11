#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <chrono>

#include <boost/multiprecision/cpp_int.hpp>

using namespace std;

namespace mp = boost::multiprecision;


mutex mtx; // mutex for output synchronization

struct factor_exponent {
    mp::cpp_int factor;
    int exponent;
};

/**
 * @brief Function to check if a number is prime
 *
 * @param n number to check if prime
 * @return true if prime, false otherwise
 */
bool isPrime(mp::cpp_int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    // if (n % 2 == 0 || n % 3 == 0) return false;
    if (n % 3 == 0) return false;
    for (mp::cpp_int i = 5; i * i <= n; i += 6) {
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
void findPrimesInRange(mp::cpp_int start, mp::cpp_int end, mp::cpp_int num, vector<factor_exponent>& primes) {

    // check all numbers in the range
    for (mp::cpp_int i = start; i <= end; i += 2) {
        
        // if i in the range is prime and num is divisible by i
        // add it to the primes vector

        if((num % i) == 0) {

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
                primes.push_back({ i, exponent });
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
vector<factor_exponent> parallelTrialDivision(mp::cpp_int num, int numThreads) {

    vector<factor_exponent> primes;
    vector<thread> threads;

    // number of threads to be used for parallelization
    // const int numThreads = thread::hardware_concurrency();

    mp::cpp_int old_num = num;

    // Checking in advance if the number is divisible by 2
    if (num % 2 == 0) {
        int exponent = 0;
        while (num % 2 == 0) {
            exponent++;
            num /= 2;
        }
        primes.push_back({ 2, exponent });
    }
    // Now the interval to check is nearly halved
    // as checking divisibility by even numbers is not needed

    cout << "Number after dividing by 2: " << num << endl;
    mp::cpp_int sqrt_num = (mp::cpp_int) mp::sqrt(num);
    cout << "Sqrt: " << sqrt_num << endl;

    // divide the work equally among the threads
    mp::cpp_int range = sqrt_num / numThreads;

    // define the start and end of the range for the first thread
    mp::cpp_int start = 3;
    mp::cpp_int end = (range % 2 == 0) ? range + 1 : range;

    // create and start the threads
    for (int i = 0; i < (numThreads - 1); ++i) {
        threads.emplace_back(findPrimesInRange, start, end, num, ref(primes));

        // Print thread identifier
        // cout <<"Thread["<<i<<"] ID: " << threads[i].get_id() << endl;

        // update the start and end for the next thread
        start = end + 2;

        if(range % 2 == 0){
            end = start + range;
        }
        else{
            end = start + range - 1;
        }
    }
    end = sqrt_num;

    // Last thread is the main thread
    findPrimesInRange(start, end, num, primes);

    // wait for the other threads to finish
    for (auto& thread : threads) {
        thread.join();
    }

    // If primes is empty (i.e. the number is prime)
    if(primes.empty()) {
        primes.push_back({ num, 1 });
    } else {

        // Check if all the factors have been found 
        // (otherwise a prime factor larger than the 
        // square root of the number is missing)

        // We have to remove from primes all the non-primes factor that were found
        // And add the (possible) missing prime factor

        cout << old_num << " = ";
        for (auto it = primes.begin(); it != primes.end(); ++it) {
            cout << it->factor << "^" << it->exponent;
            // print a * between factors except for the last one
            if (next(it) != primes.end()) {
                cout << " * ";
            }
        }
        cout << endl;


        mp::cpp_int product = 1;
        for (auto it = primes.begin(); it != primes.end(); ++it) {

            cout << "Current factor: " << it->factor << " - Current exponent: " << it->exponent << "\n";

            if(!isPrime(it->factor)) {
                cout << "NOT PRIME\n";
                // If the factor is not prime, we have to remove it
                primes.erase(it);
                it--;
                // TRY WITH THIS NUMBER 30993450745582 
                // AND NUMBER LIKE THIS ONE!
            }
            else {
                product *= pow(it->factor, it->exponent);
            }

        }
        if (product != old_num) {
            primes.push_back({ old_num / product, 1 });
        }
    }

    return primes;
}

int main(int argc, char* argv[]) {

    if (argc != 4) {
        cout << "Please provide: Number of Threads, Number to be Factorized and 0 or 1 to execute the program in BASH or USER mode." << endl;
        return 1;
    }

    // get the number of threads from the command line argument
    int NUM_THREADS = atoi(argv[1]);

    // get the number from the command line argument
    mp::cpp_int NUMBER = mp::cpp_int(argv[2]);
    cout << "Number: " << NUMBER << endl;

    // get the mode (0: bash, 1: user)
    bool EXECUTION_MODE = atoi(argv[3]);

    // start measuring time
    // auto start = chrono::high_resolution_clock::now();
    auto start = chrono::steady_clock::now();
    // chrono::steady_clock::time_point start = chrono::high_resolution_clock::now();

    // find the prime factors of the number
    vector<factor_exponent> factors = parallelTrialDivision(NUMBER, NUM_THREADS);

    // stop measuring time
    // auto end = chrono::high_resolution_clock::now();
    auto end = chrono::steady_clock::now();
    // chrono::steady_clock::time_point end = chrono::high_resolution_clock::now();


    // calculate the time duration
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
    // chrono::milliseconds duration = chrono::duration_cast<chrono::milliseconds>(end - start);

    if (EXECUTION_MODE) {
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
    }
    else {
        cout << duration.count() << endl;
    }

    return 0;
}
