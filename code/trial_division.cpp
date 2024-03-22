#include <iostream>
#include <vector>
#include <thread>
#include <mutex>

using namespace std;
mutex mtx; // Mutex per la sincronizzazione dell'output
// TO CHECK: fare l'unione delle singole strutture dati alla fine e non mettere il mutex

// Funzione per controllare se un numero è primo
bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

// Funzione per la ricerca dei divisori primi in un intervallo
void findPrimesInRange(int start, int end, int num, vector<int>& primes) {
    // scorre tutto l'intervallo
    for (int i = start; i <= end; ++i) {
        // se il i nell'intervallo è primo e se num è divisibile per i
        if (isPrime(i) && num % i == 0) {
            // aggiunta nel vettore primes
            lock_guard<mutex> lock(mtx);
            primes.push_back(i);
            // continuo a dividere finché è possibile
            // in questo modo si evita di aggiungere più 
            // volte lo stesso fattore
            while (num % i == 0) {
                num /= i;
            }
        }
    }
}

// Funzione principale per la fattorizzazione parallela
vector<int> parallelTrialDivision(int num) {
    vector<int> primes;
    vector<thread> threads;

    // Numero di thread che verranno utilizzati
    const int numThreads = thread::hardware_concurrency();
    cout << "Il numero di threads è: " << numThreads << endl;

    // Dividiamo il lavoro in parti uguali tra i thread
    int range = num / numThreads;
    int start = 2;
    int end = range;

    // Creazione e avvio dei thread
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(findPrimesInRange, start, end, num, ref(primes));
        start = end + 1;
        end = (i == numThreads - 2) ? num : start + range - 1;
    }

    // Attendo la terminazione dei thread
    for (auto& thread : threads) {
        thread.join();
    }

    return primes;
}

int main() {
    int num;
    cout << "Inserisci un numero intero da fattorizzare: ";
    cin >> num;

    vector<int> factors = parallelTrialDivision(num);

    // Stampiamo i fattori primi
    cout << "I fattori primi di " << num << " sono: ";
    for (int factor : factors) {
        cout << factor << " ";
    }
    cout << endl;

    return 0;
}
