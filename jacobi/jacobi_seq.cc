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

const int B = 1024; // the block size
const int K = 4;  // number of iterations done in one wave version; K must be even!

// A global context structure to prepare for thread parallelism
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

// compute norm of defect 
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

// simplest implementation of the Jacobi kernel
// - perform a fixed number of iterations
// - use swap to avoid copy
// - expects input in uold
// - provides output in uold
void jacobi_vanilla_kernel(int n, int iterations, double *__restrict__ uold, double *__restrict__ unew)
{
  // do iterations
  for (int i = 1; i <= iterations; i++)
  {
    for (int i1 = 1; i1 < n - 1; i1++)
      for (int i0 = 1; i0 < n - 1; i0++)
        unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    std::swap(uold, unew);
  }
}

// just calls the kernel using arguments from context structure
void jacobi_vanilla(std::shared_ptr<GlobalContext> context)
{
  jacobi_vanilla_kernel(context->n, context->iterations, context->u0, context->u1);
}

// process indices in blocks of size B in the fast direction, includes tail loops
// - perform a fixed number of iterations
// - use swap to avoid copy
// - expects input in uold
// - provides output in uold
void jacobi_blocked_kernel(int n, int iterations, double *__restrict__ uold, double *__restrict__ unew)
{
  int blocksB = ((n - 2) / B) * B; // largest number of multiple of B that fits in n-2 
 
  // do iterations
  for (int i = 0; i < iterations; i++)
  {
    int I0 = 1;
    for (; I0+B < n; I0 += B)
      for (int i1 = 1; i1 < n - 1; i1++)
        for (int i0 = I0; i0 < I0 + B; i0++)    // loop over individual indices in consecutive direction
            unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    for (int i1 = 1; i1 < n - 1; i1++)
      for (int i0 = I0; i0 < n - 1; i0++)
        unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    std::swap(uold, unew);
  }
}

// just calls the kernel using arguments from context structure
void jacobi_blocked(std::shared_ptr<GlobalContext> context)
{
  jacobi_blocked_kernel(context->n, context->iterations, context->u0, context->u1);
}

// implements wave variant computing K iterations in an overlapping fashion
// - perform a fixed number of iterations
// - use swap to avoid copy
// - expects input in uold
// - provides output in unew
void jacobi_wave_kernel(int n, int iterations, double *__restrict__ uold, double *__restrict__ unew)
{
  double *u[2];
  u[0] = uold;
  u[1] = unew;

  // do iterations
  for (int kk = 0; kk < iterations; kk += K) // we do (iterations/K) * K blocks of K iterations (possibly doing too much if K does not divide iterations)
    for (int m = 2; m <= n + K - 2; m++) // m enumerates the diagonals
      for (int k = std::max(1, m - n + 2); k <= std::min(K, m - 1); k++) // and k enumerates within a diagonal; start with 1 except when m>=n
      {
        int i1 = m - k;
        int dst = k % 2; // in principle the destination iteration index is kk+k but since kk is a multiple of K: (kk+K)%2 == kk%2
        int src = 1 - dst;
        for (int i0 = 1; i0 < n - 1; i0++)
          u[dst][i1 * n + i0] = 0.25 * (u[src][i1 * n + i0 - n] + u[src][i1 * n + i0 - 1] + u[src][i1 * n + i0 + 1] + u[src][i1 * n + i0 + n]);
      }
  // since K is even, the result is in uold = u[0]
}

void jacobi_wave(std::shared_ptr<GlobalContext> context)
{
  jacobi_wave_kernel(context->n, context->iterations, context->u0, context->u1);
}

// main function runs the experiments and outputs results as csv
int main(int argc, char **argv)
{
  std::vector<int> sizes;
  std::vector<int> iters;
  int iterations = 2048;
  for (int n = 512; n < 20000; n *= 2)
  {
    sizes.push_back(n+2);
    iters.push_back(iterations);
    iterations /= 2;
  }

  std::vector<double> performance_vanilla;
  std::vector<double> performance_blocked;
  std::vector<double> performance_wave;
  
  for (int i = 0; i < sizes.size(); i++)
  {
    int n = sizes[i];
    iterations = iters[i];

    // get global context shared by all threads
    auto context = std::make_shared<GlobalContext>(n, iterations);

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

    // vanilla
    std::fill(context->u0, (context->u0) + n*n, 0.0);
    for (int i1 = 0; i1 < n; i1++)
      for (int i0 = 0; i0 < n; i0++)
        context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
    start = get_time_stamp();
    jacobi_vanilla(context);
    stop = get_time_stamp();
    elapsed = get_duration_seconds(start, stop);
    performance_vanilla.push_back(updates / elapsed / 1e9);

    // blocked
    std::fill(context->u0, (context->u0) + n*n, 0.0);
    for (int i1 = 0; i1 < n; i1++)
      for (int i0 = 0; i0 < n; i0++)
        context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
    start = get_time_stamp();
    jacobi_blocked(context);
    stop = get_time_stamp();
    elapsed = get_duration_seconds(start, stop);
    performance_blocked.push_back(updates / elapsed / 1e9);

    // wave
    std::fill(context->u0, (context->u0) + n*n, 0.0);
    for (int i1 = 0; i1 < n; i1++)
      for (int i0 = 0; i0 < n; i0++)
        context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
    start = get_time_stamp();
    jacobi_wave(context);
    stop = get_time_stamp();
    elapsed = get_duration_seconds(start, stop);
    performance_wave.push_back(updates / elapsed / 1e9);

    // deallocate arrays
    // delete[] context->u1;
    // delete[] context->u0;
    ::operator delete[](context->u1, std::align_val_t(64));
    ::operator delete[](context->u0, std::align_val_t(64));
  }

  // print results
  std::cout << "jacobi sequential no auto vectorizer" << std::endl;
  std::cout << "N, vanilla, blocked, wave" << std::endl;
  for (int i=0; i<sizes.size(); i++)
    std::cout << sizes[i] << ", " 
              << performance_vanilla[i] 
              << ", " << performance_blocked[i] 
              << ", " << performance_wave[i] 
              << std::endl;

  return 0;
}
