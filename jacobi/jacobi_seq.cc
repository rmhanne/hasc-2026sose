#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <stdio.h>
#include "time_experiment.hh"

#ifndef __GNUC__
#define __restrict__
#endif

const int B = 20; // the block size
const int K = 4;  // number of iterations done in one wave version

struct GlobalContext
{
  // input data
  int n;          // nxn lattice of points including boundary
  int iterations; // number of iterations to do
  double *u0;     // the initial guess
  double *u1;     // temporary vector

  // output data

  GlobalContext(int n_)
      : n(n_)
  {
  }
  GlobalContext(int n_, int iterations_)
      : n(n_), iterations(iterations_)
  {
  }
};

double defect_norm(int n, double *__restrict__ u)
{
  double sum = 0.0;
  for (int i1 = 1; i1 < n - 1; i1++)
    for (int i0 = 1; i0 < n - 1; i0++)
    {
      double d = 4.0 * u[i1 * n + i0] - (u[i1 * n + i0 - n] + u[i1 * n + i0 - 1] + u[i1 * n + i0 + 1] + u[i1 * n + i0 + n]);
      sum += d * d;
    }
  return sqrt(sum);
}

void jacobi_vanilla_kernel(int n, int iterations, double *__restrict__ uold, double *__restrict__ unew)
{
  auto d0 = defect_norm(n,uold);
  auto last=d0;
  int iter;

  // do iterations
  for (int i = 1; i <= iterations; i++)
  {
    for (int i1 = 1; i1 < n - 1; i1++)
      for (int i0 = 1; i0 < n - 1; i0++)
        unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    std::swap(uold, unew);
  }
}

void jacobi_vanilla(std::shared_ptr<GlobalContext> context)
{
  jacobi_vanilla_kernel(context->n, context->iterations, context->u0, context->u1);
}

void jacobi_wave_kernel(int n, int iterations, double *__restrict__ uold, double *__restrict__ unew)
{
  double *u[2];
  u[0] = uold;
  u[1] = unew;

  // do iterations
  for (int kk = 0; kk < iterations; kk += K)
    for (int m = 2; m <= n + K - 2; m++)
      for (int k = std::max(1, m - n + 2); k <= std::min(K, m - 1); k++)
      {
        int i1 = m - k;
        int dst = k % 2;
        int src = 1 - dst;
        for (int i0 = 1; i0 < n - 1; i0++)
          u[dst][i1 * n + i0] = 0.25 * (u[src][i1 * n + i0 - n] + u[src][i1 * n + i0 - 1] + u[src][i1 * n + i0 + 1] + u[src][i1 * n + i0 + n]);
      }
}

void jacobi_wave(std::shared_ptr<GlobalContext> context)
{
  jacobi_wave_kernel(context->n, context->iterations, context->u0, context->u1);
}

void jacobi_blocked(std::shared_ptr<GlobalContext> context)
{
  int n = context->n;
  double *uold = context->u0;
  double *unew = context->u1;

  int blocksB = ((n - 2) / B) * B; // number of indices in blocks of size B per direction, excluding boundary 
 
  // do iterations
  for (int i = 0; i < context->iterations; i++)
  {
    for (int I1 = 1; I1 < 1 + blocksB; I1 += B)   // loop over first index in each block
      for (int I0 = 1; I0 < 1 + blocksB; I0 += B) // same in the first direction
        for (int i1 = I1; i1 < I1 + B; i1++)      // loop over individual indices in slow direction
          for (int i0 = I0; i0 < I0 + B; i0++)    // loop over individual indices in consecutive direction
            unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    for (int I0 = 1; I0 < 1 + blocksB; I0 += B)
      for (int i1 = 1 + blocksB; i1 < n - 1; i1++)
        for (int i0 = I0; i0 < I0 + B; i0++)
          unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    for (int i1 = 1 + blocksB; i1 < n - 1; i1++)
      for (int i0 = 1 + blocksB; i0 < n - 1; i0++)
        unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    std::swap(uold, unew);
  }

  // result should be in u1
  if (context->u1 != uold)
    std::swap(context->u0, context->u1);
}

// main function runs the experiments and outputs results as csv
int main(int argc, char **argv)
{
  // read parameters
  int n = 1024;
  int iterations = 1000;
  if (argc != 3)
  {
    std::cout << "usage: ./jacobi_seq <size> <iterations>"
              << std::endl;
    exit(1);
  }
  sscanf(argv[1], "%d", &n);
  sscanf(argv[2], "%d", &iterations);
  std::cout << "jacobi_vanilla: n=" << n
          << " iterations=" << iterations
          << " memory (mbytes)=" << (n*n)*8.0*2.0/1024.0/1024.0
          << std::endl;

  // check sizes
  if (K % 2 == 1)
  {
    std::cout << "K must be even" << std::endl;
    exit(1);
  }
  if (iterations % K != 0)
  {
    std::cout << "iterations must be a multiple of K=" << K << std::endl;
    exit(1);
  }

  // get global context shared by all threads
  auto context = std::make_shared<GlobalContext>(n,iterations);

  // allocate aligned arrays
  context->u0 = new (std::align_val_t(64)) double[n * n];
  context->u1 = new (std::align_val_t(64)) double[n * n];

  // fill boundary values and initial values
  auto g = [&](int i0, int i1)
  { return (i0 > 0 && i0 < n - 1 && i1 > 0 && i1 < n - 1)
               ? 0.0
               : ((double)(i0 + i1)) / n; };

  // warmup
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  auto start = get_time_stamp();
  jacobi_vanilla(context);
  auto stop = get_time_stamp();
  double elapsed = get_duration_seconds(start, stop);
  double updates = context->iterations;
  updates *= (n - 2) * (n - 2);

  std::cout << "N,";
  std::cout << "vanilla,";
  std::cout << "blocked,";
  std::cout << "wave,";
  std::cout << std::endl;
  std::cout << n * n;

  // vanilla
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  start = get_time_stamp();
  jacobi_vanilla(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start, stop);
  std::cout << "," << updates / elapsed / 1e9;

  
  // blocked
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  start = get_time_stamp();
  jacobi_blocked(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start, stop);
  std::cout << "," << updates / elapsed / 1e9;

  // wave
  for (int i1 = 0; i1 < n; i1++)
    for (int i0 = 0; i0 < n; i0++)
      context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  start = get_time_stamp();
  jacobi_wave(context);
  stop = get_time_stamp();
  elapsed = get_duration_seconds(start, stop);
  std::cout << "," << updates / elapsed / 1e9;
  std::cout << std::endl;

  // deallocate arrays
  //delete[] context->u1;
  //delete[] context->u0;
  ::operator delete[] (context->u1, std::align_val_t(64));
  ::operator delete[] (context->u0, std::align_val_t(64));

  return 0;
}
