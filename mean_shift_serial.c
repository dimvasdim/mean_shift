/**
    Author: Dimitriadis Vasileios 8404
    Faculty of Electrical and Computer Engineering AUTH
    3rd assignment at Parallel and Distributed Systems (7th semester)
    This is a serial implementation of mean shift algorithm using the
    Gaussian probability density function.
  **/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#define N 60000
#define DIMENSIONS 30
#define EPSILON 0.001
#define VAR 0.001 // =σ^2 variance

struct timeval startwtime, endwtime;
double seq_time;

double **alloc_2d_double(int rows, int cols);
void free_d(double **a);
void getinput(double **x, char *filename);
void copy_matrix(double **a, double **b);
void copy_array(double **a, double **b, int i);
void find_new_y (double **x, double **y, double **y_new, int i);
double find_distance(double **y, int i, double **x, int j);
int check_converge(double **y, double **y_new, int i);
void show_results(double **y_new);

int main(int argc, char **argv)
{
  int converge, i, j;

  if (argc != 2)
  {
    printf("Need as input a dataset to process\n");
    exit (1);
  }

  double **x = alloc_2d_double(N, DIMENSIONS);
  getinput(x, argv[1]);

  double **y = alloc_2d_double(N, DIMENSIONS);
  copy_matrix(x, y);

  double **y_new = alloc_2d_double(N, DIMENSIONS);

  gettimeofday (&startwtime, NULL);
  for (i=0; i<N; i++)
  {
    converge = 0;
    while (!converge)
    {
      find_new_y (x, y, y_new, i);
      converge = check_converge(y, y_new, i);
      copy_array(y_new, y, i);
    }
  }
  gettimeofday (&endwtime, NULL);
  seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6
                        + endwtime.tv_sec - startwtime.tv_sec);
  printf("Time needed for mean shift is %f sec\n", seq_time);
  show_results(y_new);
  free_d(x);
  free_d(y);
  free_d(y_new);

  return (0);
}

void getinput(double **x, char *filename)
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
  str = fgets(str, 2 * DIMENSIONS * sizeof(double), fin);
  while (str != NULL && i < N)
  {
    token = strtok(str, "\t"); //get one dimension per recursion.
    j = 0;
    while (token != NULL && j < DIMENSIONS)
    {
      x[i][j] = atof(token);
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

void copy_matrix(double **a, double **b)
{
  for (int i=0; i<N; i++)
  {
    for (int j=0; j<DIMENSIONS; j++)
    {
      b[i][j] = a[i][j];
    }
  }
}

void copy_array(double **a, double **b, int i)
{
  for (int j=0; j<DIMENSIONS; j++)
  {
    b[i][j] = a[i][j];
  }
}

void find_new_y (double **x, double **y, double **y_new, int i)
{
  int j, l;
  double distance, nominator[DIMENSIONS], denominator, k;
  for (l=0; l<DIMENSIONS; l++)
  {
    nominator[l] = 0;
  }
  denominator = 0;
  for (j=0; j<N; j++)
  {
    distance = find_distance(y, i, x, j);
    if (sqrt(distance) <= VAR)
    {
      k = exp(-distance / (2 * VAR));
      for (l=0; l<DIMENSIONS; l++)
      {
        nominator[l] = nominator[l] + (k * x[j][l]);
      }
      denominator = denominator + k;
    }
  }
  for (l=0; l<DIMENSIONS; l++)
  {
    y_new[i][l] = nominator[l] / denominator;
  }
}


double find_distance(double **y, int i, double **x, int j)
{
  double distance = 0;
  for (int l=0; l<DIMENSIONS; l++)
  {
    distance = distance + pow(y[i][l]-x[j][l], 2);
  }
  return distance;
}

int check_converge(double **y, double **y_new, int i)
{
  int ret = 0;
  double distance = find_distance(y_new, i, y, i);
  if (sqrt(distance) < EPSILON)
  {
    ret = 1;
  }
  return (ret);
}

double **alloc_2d_double(int rows, int cols)
{
  double *data = (double *)malloc(rows * cols * sizeof(double));
  if (data == NULL)
  {
    printf("Failed to allocate data\n");
    exit(1);
  }
  double **array= (double **)malloc(rows * sizeof(double*));
  if (array == NULL)
  {
    printf("Failed to allocate array\n");
    exit(1);
  }
  for (int i=0; i<rows; i++)
  {
    array[i] = &(data[cols*i]);
  }
  return array;
}

void free_d(double **a)
{
  free(a[0]);
  free(a);
}

void show_results(double **y_new)
{
  int i,j;
  for(i=0; i<20; i++)
  {
    for (j=0; j<DIMENSIONS; j++)
    {
      printf("%f ", y_new[i][j]);
    }
    printf("\n");
  }
}
