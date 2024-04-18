# Slide 1

Il progetto che Giovanni ed io stiamo presentando si concentra sulla fattorizzazione dei numeri primi che consiste nell'esprimere un numero intero come il prodotto di numeri primi. Questo processo non solo riveste importanza teorica, ma ha anche numerose applicazioni pratiche, tra cui la crittografia e la sicurezza informatica.

# Slide 2

Esistono diversi algoritmi per raggiungere l'obiettivo della fattorizzazione dei numeri primi, ma abbiamo optato per l'utilizzo del metodo del trial division, poiché è efficiente e può essere facilmente parallelizzato.

Per iniziare, consideriamo la versione seriale di questo algoritmo, che prende in input un numero intero N. 
Si esegue quindi un ciclo for, partendo da i=2 (poiché 1 non è un numero primo) e arrivando fino alla radice quadrata di N. 
Durante questo ciclo, controlliamo se i divide N. 
In caso affermativo, dividiamo N per i e aggiungiamo i al vettore dei fattori primi. 
Al termine del ciclo, se N è diverso da 1, significa che è esso stesso un numero primo e quindi lo aggiungiamo al vettore dei fattori primi.

# Slide 3

Qui abbiamo riportato un esempio col numero 140

# Slide 4

Avviando il ciclo, osserviamo che il numero 140 viene diviso due volte per 2.
# Slide 5

Pertanto, aggiungiamo il numero 2 al vettore dei numeri primi e il numero 140 viene ridotto a 35.

# Slide 6

Alla fine del ciclo, ci troviamo con un numero ridotto a 1, il che significa che non dobbiamo aggiungerlo al vettore e nel vettore Primes troviamo la sua fattorizzazione.

# Slide 8

Nella versione parallela dell'algoritmo, manteniamo lo stesso principio di base dell'algoritmo seriale, ma distribuiamo il carico di lavoro su diversi thread. 
Notiamo che abbiamo riservato l'ultima porzione dell'intervallo di numeri da fattorizzare al thread principale. Questa scelta è stata fatta per evitare che il thread principale rimanesse inattivo mentre attende i risultati degli altri thread.

Inoltre, è stato necessario introdurre un Mutex per gestire l'accesso al vettore dei numeri primi. È importante notare che, nonostante l'aggiunta del Mutex, non abbiamo riscontrato problemi di sincronizzazione. Questo è dovuto al fatto che gli accessi al vettore dei numeri primi sono relativamente pochi durante l'esecuzione dell'algoritmo parallelo.

# Slide 9

Nel contesto del nostro progetto, ci siamo prefissati due obiettivi. 
Il primo obiettivo è verso l'utente e si concentra sull'ottenere un tempo di esecuzione inferiore a un secondo per numeri con 18 cifre decimali.

Il secondo obiettivo è rivolto allo sviluppatore e mira a ottenere un cumulative speedup pari almeno alla metà dei core logici del sistema. 

# Slide 10

Per i nostri esperimenti, abbiamo utilizzato un processore Intel i5 dotato di 6 core fisici e 2 thread per core, per un totale di 12 core logici. 

Qui sono riportate le dimensioni della cache utilizzata. Per quanto riguarda la cache L1 e L2, i valori forniti sono cumulativi.

# Slide 11

*Da vedere se modificarla come la vuole Di Tecco*

# Slide 12

Abbiamo sviluppato un primo algoritmo e lo abbiamo sottoposto a test utilizzando un numero a 18 cifre, in linea con i nostri obiettivi. 
Purtroppo, i risultati ottenuti sono stati al di fuori delle nostre aspettative, con un tempo di esecuzione che si avvicina ai 9 minuti. 

# Slide 13

Siamo quindi andati a studiare le performance nel dettaglio.

# Slide 14

In questa slide, abbiamo riportato i tempi di esecuzione dell'algoritmo nella sua prima versione in relazione al numero di thread utilizzati.

Abbiamo osservato una discesa più marcata fino a 6 thread, il che riflette il numero di core fisici presenti nel sistema. 
Successivamente, abbiamo registrato un secondo andamento fino a 12 thread, in linea con il numero di thread logici disponibili. 
Infine, il tempo di esecuzione si è stabilizzato.

# Slide 15

In questa slide, abbiamo presentato il cumulative speedup dell'algoritmo in base al numero di thread utilizzati. 
Abbiamo anche in questo caso osservato tre distinti andamenti. 
Inizialmente, abbiamo registrato un aumento più marcato fino a 6 thread, seguito da un secondo incremento da 7 a 12 thread. Infine, il cumulative speedup si è stabilizzato.

*Fine parte Alice*
# Slide 17

