Consigli per testare l'Algoritmo scelto:

- Variare il numero di Thread: dopo il numero di Thread logici disponibili sul nostro processore è inutile andare avanti di molto, perché l'overhead introdotto supera il beneficio che ne deriva.
- Variare il carico: il punto di saturazione si raggiunge quando le prestazioni non migliorano più.

Studiare la scalabilità dell'Algoritmo e presentare al Professore risultati concreti con grafici di facile lettura.

---

Da fare:

- Aggiungere controllo argc se è maggiore di due
- Vedere la modifica con struttura con numero ed esponente e modificare findPrimeInRange
- Per futuri grafici mettere più thread per vedere che non migliora (hardware + hardware/2)
- Fare ottimizzazione range: sqrt(N) - 3