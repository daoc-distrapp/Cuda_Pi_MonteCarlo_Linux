#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

static const long BLOCKS = 256;
static const long THREAD_X_BLOCK = 256;
static const long ITER_X_THREAD = 1024 * 64;

__global__
void piMC(long long *blockCounter) { // blockCounter debe tener un contador por cada bloque

	// Debe haber un contador por cada hilo en el bloque (compartido  en el bloque)
	__shared__ long long threadCounter[THREAD_X_BLOCK];

	// ID de la thread
	int id = threadIdx.x + blockIdx.x * blockDim.x;

	// Inicializa el RNG
	curandState_t rng;
	curand_init(clock64(), id, 0, &rng);

	// Inicializa el contador
	threadCounter[threadIdx.x] = 0;

	// Calcula los puntos dentro del círculo
	for (int i = 0; i < ITER_X_THREAD; i++) {
		float x = curand_uniform(&rng);
		float y = curand_uniform(&rng);
		threadCounter[threadIdx.x] += 1 - int(x * x + y * y);
	}

	// La primera thread en cada bloque suma los contadores individuales en el de bloque
	if (threadIdx.x == 0) {
		// Inicializa el contador de este bloque
		blockCounter[blockIdx.x] = 0;
		// Suma los contadores de thread en el de bloque
		for (int i = 0; i < THREAD_X_BLOCK; i++) {
			blockCounter[blockIdx.x] += threadCounter[i];
		}
	}
}

int main(void) {
	// Crea el buffer para los contadores de bloque en el host
	long long *blockCounter = (long long*)malloc(sizeof(long long) * BLOCKS);

	// Crea el buffer para los contadores de bloque en la GPU
	long long *gpuBlockCounter;
	cudaMalloc(&gpuBlockCounter, sizeof(long long) * BLOCKS);

	// Ejecuta la kernel
	piMC<<<BLOCKS, THREAD_X_BLOCK>>>(gpuBlockCounter);

	// Recupera el resultado desde la GPU y lo pone en el buffer del host
	cudaMemcpy(blockCounter, gpuBlockCounter, sizeof(long long) * BLOCKS, cudaMemcpyDeviceToHost);

	// Suma los contadores y calcula PI
	long long total = 0;
	for (int i = 0; i < BLOCKS; i++) {
		total += blockCounter[i];
	}
	long long iters = BLOCKS * THREAD_X_BLOCK * ITER_X_THREAD;
	printf("Aproximado con %lld iteraciones\n", iters);
	printf("%lld puntos dentro del círculo\n", total);
	printf("PI= %f\n", 4.0 * (double)total/(double)iters);

	// Libera los recursos
	cudaFree(gpuBlockCounter);
	free(blockCounter);

	return 0;
}
