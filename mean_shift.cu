/**
    Author: Dimitriadis Vasileios 8404
    Faculty of Electrical and Computer Engineering AUTH
    3rd assignment at Parallel and Distributed Systems (7th semester)
    This is a parallel implementation of mean shift algorithm using the
    Gaussian probability density function.
  **/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 60000
#define DIMENSIONS 5
#define EPSILON 0.001
#define VAR 0.001 // =σ^2 variance
#define N_Threads 1024

struct timeval startwtime, endwtime;
double seq_time;


void getinput(double *x, char *filename);
__global__ void meanshift(double *dev_x, double *dev_y, int dim, double eps, double var);
__device__ double find_distance(double *y, int i, double *x, int j, int dim);
void show_results(double *y_new);

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    printf("Need as input a dataset to process\n");
    exit (1);
  }

  double *x = (double *)malloc(N * DIMENSIONS * sizeof(double));
  if (x == NULL)
  {
    printf("Failed to allocate data at x...\n");
    exit(1);
  }
  getinput(x, argv[1]);

  double *y = (double *)malloc(N * DIMENSIONS * sizeof(double));
  if (y == NULL)
  {
    printf("Failed to allocate data at y...\n");
    exit(1);
  }

  double *dev_x;
  cudaMalloc(&dev_x, N * DIMENSIONS * sizeof(double));

  double *dev_y;
  cudaMalloc(&dev_y, N * DIMENSIONS * sizeof(double));

  cudaMemcpy(dev_x, x, N * DIMENSIONS * sizeof(double), cudaMemcpyHostToDevice);

  //Initialize y as x in gpu.
  cudaMemcpy(dev_y, x, N * DIMENSIONS * sizeof(double), cudaMemcpyHostToDevice);

  cudaError_t error;
  size_t shared_size = N_Threads * DIMENSIONS + N_Threads;
  gettimeofday (&startwtime, NULL);
  meanshift<<<N, N_Threads, sizeof(double) * shared_size>>>(dev_x, dev_y, DIMENSIONS, EPSILON, VAR);
  gettimeofday (&endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
                        + endwtime.tv_sec - startwtime.tv_sec);

  cudaMemcpy(y, dev_y, N * DIMENSIONS * sizeof(double), cudaMemcpyDeviceToHost);
  error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    printf("Error at copying back: %s\n", cudaGetErrorString(error));
    exit(1);
  }
  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess)
  {
    printf("Error at Sync: %s\n", cudaGetErrorString(error));
    exit(1);
  }
  printf("Time needed for mean shift is %f sec\n", seq_time);
  show_results(y);
  free(x);
  free(y);
  cudaFree(dev_x);
  cudaFree(dev_y);

  return (0);
}

void getinput(double *x, char *filename)
{
  FILE *fin;
  int i = 0, j;
  char *str = (char *)malloc(2 * DIMENSIONS * sizeof(double));
  char *token = (char *)malloc(sizeof(double));
  fin = fopen(filename, "r");
  if (fin == NULL)
  {
    printf("Error opening the file...");
    exit(1);
  }
  str = fgets(str, 2 * DIMENSIONS * sizeof(double), fin); //Take one point.
  while (str != NULL && i < N)
  {
    token = strtok(str, "\t"); //get one dimension per recursion.
    j = 0;
    while (token != NULL && j < DIMENSIONS)
    {
      x[i*DIMENSIONS + j] = atof(token);
      token = strtok(NULL, "\t");
      j++;
    }
    str = fgets(str, 2 * DIMENSIONS * sizeof(double), fin);
    i++;
  }
  fclose(fin);
  free(str);
  free(token);
}


__global__
void meanshift(double *dev_x, double *dev_y, int dim, double eps, double var)
{
  int start, end;
  // Every block is finding the new y until convergence.
  int i = blockIdx.x;
  int j = threadIdx.x;
  int n = gridDim.x;
  int n_th = blockDim.x;
  /** Every thread is processing a chunk of the data in order
      to find distances between y_i and all x faster. If the
      number of elements is devided equally by the number of
      threads then the chunk is N/(# of Blocks). If it is not then
      the first N%(# of Blocks) have one more element to process.
  **/
  int chunk = n  / n_th;
  if ((n % n_th) != 0)
  {
    if (j < (n % n_th))
    {
      chunk = chunk + 1;
      start = chunk * j;
      end = start + chunk;
    }
    else
    {
      start = chunk * j + (n % n_th);
      end = start + chunk;
    }
  }
  else
  {
    start = chunk * j;
    end = start + chunk;
  }
  /** Each block has its own shared memory and the
      size of it is number of threads multiplied by
      (dimensions + 1) to store the values of nominators
       and denominator that each thread finds.
  **/
  extern __shared__ double s[];
  double *nominator = &s[0];
  double *denominator = &s[n_th * dim];
  __shared__ int converge;
  converge = 0;
  double distance = 0, k;
  int l, r;
  while (!converge)
  {
    //Initialize nominators and denominators as 0.
    for (r=0; r<dim; r++)
    {
      nominator[j*dim + r] = 0;
    }
    denominator[j] = 0;
    // Every thread is responsible of finding the new nominators
    // and denominator in it's chunk.
    for (l=start; l<end; l++)
    {
      distance = find_distance(dev_y, i, dev_x, l, dim);
      if (sqrt(distance) <= var)
      {
        k = exp(-distance / (2 * var)); //Guassian possibility density function.
      }
      else
      {
        k = 0;
      }
      for (r=0; r<dim; r++)
      {
        nominator[j*dim + r] += k * dev_x[l*dim + r];
      }
      denominator[j] += k;
    }
    __syncthreads();
    // Reduction
    for (l=n_th/2; l>0; l>>=1)
    {
      if (j < l)
      {
        for (r=0; r<dim; r++)
        {
          nominator[j*dim + r] += nominator[(j+l) * dim + r];
        }
        denominator[j] += denominator[j+l];
      }
      __syncthreads();
    }
    // Threads from 0 to dim-1 store in the first column
    // of nominator the values of new y
    if (j < dim)
    {
      nominator[j] = nominator[j] / denominator[0];
    }
    __syncthreads();
    // Only first thread checking the converge.
    if (j == 0)
    {
      distance = 0;
      for (r=0; r<dim; r++)
      {
        distance += pow(dev_y[i*dim + r] - nominator[r], 2);
      }
      if (sqrt(distance) < eps)
      {
        converge = 1;
      }
    }
    __syncthreads();
    // New y is stored in place of the previous y.
    if (j < dim)
    {
      dev_y[i*dim + j] = nominator[j];
    }
    __syncthreads();
  }
}

__device__
double find_distance(double *y, int i, double *x, int j, int dim)
{
  double distance = 0;
  for (int l=0; l<dim; l++)
  {
    distance = distance + pow(y[i*dim + l]-x[j*dim + l], 2);
  }
  return distance;
}

void show_results(double *y_new)
{
  int i,j;
  for(i=0; i<20; i++)
  {
    for (j=0; j<DIMENSIONS; j++)
    {
      printf("%f ", y_new[i*DIMENSIONS + j]);
    }
    printf("\n");
  }
}
