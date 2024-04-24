# Slide 1

Il progetto che Giovanni ed io stiamo presentando si concentra sulla fattorizzazione dei numeri primi che consiste nell'esprimere un numero intero come il prodotto di numeri primi. Questo processo non solo riveste importanza teorica, ma ha anche numerose applicazioni pratiche, tra cui la crittografia e la sicurezza informatica.

---
The project that Giovanni and I are presenting focuses on prime factorization, which involves expressing an integer as the product of prime numbers. This process not only holds theoretical importance but also has numerous practical applications, including cryptography and cybersecurity.

# Slide 2

Esistono diversi algoritmi per raggiungere l'obiettivo della fattorizzazione dei numeri primi, ma abbiamo optato per l'utilizzo del metodo del trial division, poiché è efficiente e può essere facilmente parallelizzato.

Per iniziare, consideriamo la versione seriale di questo algoritmo, che prende in input un numero intero N. 
Si esegue quindi un ciclo for, partendo da i=2 (poiché 1 non è un numero primo) e arrivando fino alla radice quadrata di N. 
Durante questo ciclo, controlliamo se i divide N. 
In caso affermativo, dividiamo N per i e aggiungiamo i al vettore dei fattori primi. 
Al termine del ciclo, se N è diverso da 1, significa che è esso stesso un numero primo e quindi lo aggiungiamo al vettore dei fattori primi.

---
There are several algorithms to achieve prime factorization, but we have opted for using the trial division method as it is efficient and can be easily parallelized.

To begin, let's consider the serial version of this algorithm, which takes an integer N as input. We then execute a for loop, starting from i=2 (since 1 is not a prime number) and ending at the square root of N. During this loop, we check if i divides N. If it does, we divide N by i and add i to the vector of prime factors. At the end of the loop, if N is different from 1, it means it is itself a prime number, so we add it to the vector of prime factors.

# Slide 3

Qui abbiamo riportato un esempio col numero 140.

---
Here, we have provided an example with the number 140.

# Slide 4

Avviando il ciclo, osserviamo che il numero 140 viene diviso due volte per 2.

---
Starting the loop, we observe that the number 140 is divided twice by 2.

# Slide 5

Pertanto, aggiungiamo il numero 2 al vettore dei numeri primi e il numero 140 viene ridotto a 35.

---
Therefore, we add the number 2 to the vector of prime numbers, and the number 140 is reduced to 35.

# Slide 6

Alla fine del ciclo, ci troviamo con un numero ridotto a 1, il che significa che non dobbiamo aggiungerlo al vettore e nel vettore Primes troviamo la sua fattorizzazione.

---
At the end of the loop, we find that the number is reduced to 1, which means we don't need to add it to the vector, and in the Primes vector, we find its factorization.

# Slide 8

Nella versione parallela dell'algoritmo, manteniamo lo stesso principio di base dell'algoritmo seriale, ma distribuiamo il carico di lavoro su diversi thread. 
Notiamo che abbiamo riservato l'ultima porzione dell'intervallo di numeri da fattorizzare al thread principale. Questa scelta è stata fatta per evitare che il thread principale rimanesse inattivo mentre attende i risultati degli altri thread.

Inoltre, è stato necessario introdurre un Mutex per gestire l'accesso al vettore dei numeri primi. È importante notare che, nonostante l'aggiunta del Mutex, non abbiamo riscontrato problemi di sincronizzazione. Questo è dovuto al fatto che gli accessi al vettore dei numeri primi sono relativamente pochi durante l'esecuzione dell'algoritmo parallelo.

---
In the parallel version of the algorithm, we maintain the same basic principle as the serial algorithm but distribute the workload across multiple threads. We note that we have allocated the last portion of the range of numbers to be factored to the main thread. This choice was made to prevent the main thread from becoming idle while waiting for the results of the other threads.

Additionally, we needed to introduce a Mutex to handle access to the vector of prime numbers. It's important to note that, despite adding the Mutex, we did not encounter synchronization issues. This is because accesses to the vector of prime numbers are relatively few during the execution of the parallel algorithm.

# Slide 9

Nel contesto del nostro progetto, ci siamo prefissati due obiettivi. 
Il primo obiettivo è verso l'utente e si concentra sull'ottenere un tempo di esecuzione inferiore a un secondo per numeri con 18 cifre decimali.

Il secondo obiettivo è rivolto allo sviluppatore e mira a ottenere un cumulative speedup pari almeno alla metà dei core logici del sistema. 

---
In the context of our project, we have set two objectives.

The first objective is user-focused and aims to achieve an execution time of less than one second for numbers with 18 decimal digits.

The second objective is developer-focused and aims to achieve a cumulative speedup of at least half of the logical cores of the system.

# Slide 10

Per i nostri esperimenti, abbiamo utilizzato un processore Intel i5 dotato di 6 core fisici e 2 thread per core, per un totale di 12 core logici. 

Qui sono riportate le dimensioni della cache utilizzata. Per quanto riguarda la cache L1 e L2, i valori forniti sono cumulativi.

---
For our experiments, we used an Intel i5 processor with 6 physical cores and 2 threads per core, totaling 12 logical cores.

Here are the sizes of the cache used. Regarding L1 and L2 cache, the values provided are cumulative.

# Slide 11

*Da vedere se modificarla come la vuole Di Tecco*

# Slide 12

Abbiamo sviluppato un primo algoritmo e lo abbiamo sottoposto a test utilizzando un numero a 18 cifre, in linea con i nostri obiettivi. 
Purtroppo, i risultati ottenuti sono stati al di fuori delle nostre aspettative, con un tempo di esecuzione che si avvicina ai 9 minuti. 

---
We developed a preliminary algorithm and tested it using an 18-digit number, in line with our objectives. Unfortunately, the results obtained were beyond our expectations, with an execution time approaching 9 minutes.

# Slide 13

Siamo quindi andati a studiare le performance nel dettaglio.

---
So, we went to study the performance in detail.

# Slide 14

In questa slide, abbiamo riportato i tempi di esecuzione dell'algoritmo nella sua prima versione in relazione al numero di thread utilizzati.

Abbiamo osservato una discesa più marcata fino a 6 thread, il che riflette il numero di core fisici presenti nel sistema. 
Successivamente, abbiamo registrato un secondo andamento fino a 12 thread, in linea con il numero di thread logici disponibili. 
Infine, il tempo di esecuzione si è stabilizzato.

---
In this slide, we have reported the execution times of the algorithm in its first version in relation to the number of threads used.

We observed a more pronounced descent up to 6 threads, reflecting the number of physical cores present in the system. Subsequently, we recorded a second trend up to 12 threads, in line with the number of logical threads available. Finally, the execution time stabilized.

# Slide 15

In questa slide, abbiamo presentato il cumulative speedup dell'algoritmo in base al numero di thread utilizzati. 
Abbiamo anche in questo caso osservato tre distinti andamenti. 
Inizialmente, abbiamo registrato un aumento più marcato fino a 6 thread, seguito da un secondo incremento da 7 a 12 thread. Infine, il cumulative speedup si è stabilizzato.

---
In this slide, we presented the cumulative speedup of the algorithm based on the number of threads used. Once again, we observed three distinct trends. Initially, we recorded a more pronounced increase up to 6 threads, followed by a second increment from 7 to 12 threads. Finally, the cumulative speedup stabilized.

*Fine parte Alice*

# Slide 17

