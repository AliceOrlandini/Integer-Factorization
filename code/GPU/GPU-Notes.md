Really useful tutorial on CUDA programming
https://www.youtube.com/watch?v=xwbD6fL5qC8

CUDA Atomic Operations thread or block level?
https://stackoverflow.com/a/57115601



Note per GPU:
Provare ad usare numero di blocchi quanto è il numero di SM nella GPU. iN caso provare anche ad andare oltre e vedere come è l'impatto sulle prestazioni...
Impossibile superare 1024 threads per block
In un warp ci sono 32 threads. 
DIvergence accade solo dentro warp
Diversi warp eseguono in modo indipendente
warp-to-warp context switch is free!
