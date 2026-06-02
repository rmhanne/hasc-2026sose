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

#ifdef _OPENMP
#include <omp.h> // headers for runtime if available
#endif

#include <experimental/simd>
namespace stdx = std::experimental;

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
#pragma omp parallel for schedule(static)
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

// simplest implementation of the Jacobi kernel
// - perform a fixed number of iterations
// - use swap to avoid copy
// - expects input in uold
// - provides output in uold
void jacobi_omp_stdsimd_kernel(int n, int iterations, double *__restrict__ uold, double *__restrict__ unew)
{
  using simd_t = stdx::native_simd<double>;
  constexpr int simd_width = simd_t::size();

  // do iterations
  simd_t q;
  q = 0.25;
  simd_t s0, s1, s2, s3;
  simd_t t0, t1, t2, t3;

  for (int i = 1; i <= iterations; i++)
  {
    // do rows
#pragma omp parallel for schedule(static) private(q, s0, s1, s2, s3, t0, t1, t2, t3)
    for (int i1 = 1; i1 < n - 1; i1++)
    {
      // sweep over one row
      int i0 = 1;
      for (; i0 + 4 * simd_width < n; i0 += 4 * simd_width)
      {
        int index = i1 * n + i0;
        s0.copy_from(&uold[index - n], stdx::element_aligned);
        s1.copy_from(&uold[index - n + simd_width], stdx::element_aligned);
        s2.copy_from(&uold[index - n + 2 * simd_width], stdx::element_aligned);
        s3.copy_from(&uold[index - n + 3 * simd_width], stdx::element_aligned);
        s0 *= q;
        s1 *= q;
        s2 *= q;
        s3 *= q;
        t0.copy_from(&uold[index - 1], stdx::element_aligned);
        t1.copy_from(&uold[index - 1 + simd_width], stdx::element_aligned);
        t2.copy_from(&uold[index - 1 + 2 * simd_width], stdx::element_aligned);
        t3.copy_from(&uold[index - 1 + 3 * simd_width], stdx::element_aligned);
        s0 += q * t0;
        s1 += q * t1;
        s2 += q * t2;
        s3 += q * t3;
        t0.copy_from(&uold[index + 1], stdx::element_aligned);
        t1.copy_from(&uold[index + 1 + simd_width], stdx::element_aligned);
        t2.copy_from(&uold[index + 1 + 2 * simd_width], stdx::element_aligned);
        t3.copy_from(&uold[index + 1 + 3 * simd_width], stdx::element_aligned);
        s0 += q * t0;
        s1 += q * t1;
        s2 += q * t2;
        s3 += q * t3;
        t0.copy_from(&uold[index + n], stdx::element_aligned);
        t1.copy_from(&uold[index + n + simd_width], stdx::element_aligned);
        t2.copy_from(&uold[index + n + 2 * simd_width], stdx::element_aligned);
        t3.copy_from(&uold[index + n + 3 * simd_width], stdx::element_aligned);
        s0 += q * t0;
        s1 += q * t1;
        s2 += q * t2;
        s3 += q * t3;
        s0.copy_to(&unew[index], stdx::element_aligned);
        s1.copy_to(&unew[index + simd_width], stdx::element_aligned);
        s2.copy_to(&unew[index + 2 * simd_width], stdx::element_aligned);
        s3.copy_to(&unew[index] + 3 * simd_width, stdx::element_aligned);
      }
      for (; i0 < n - 1; i0++)
        unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    }
    std::swap(uold, unew);
  }
}

// just calls the kernel using arguments from context structure
void jacobi_omp_stdsimd(std::shared_ptr<GlobalContext> context)
{
  jacobi_omp_stdsimd_kernel(context->n, context->iterations, context->u0, context->u1);
}

// simplest implementation of the Jacobi kernel
// - perform a fixed number of iterations
// - use swap to avoid copy
// - expects input in uold
// - provides output in uold
void jacobi_omp_stdsimd_kernel2(int n, int iterations, double *__restrict__ uold, double *__restrict__ unew)
{
  using simd_t = stdx::native_simd<double>;
  constexpr int simd_width = simd_t::size();

  // do iterations
  simd_t q;
  q = 0.25;
  simd_t s0, s1, s2, s3;
  simd_t t0, t1, t2, t3, t4, t5;
  simd_t u0, u1, u2, u3;

  for (int i = 1; i <= iterations; i++)
  {
    // do rows
#pragma omp parallel for schedule(static) private(q, s0, s1, s2, s3, t0, t1, t2, t3,t4,t5,u0,u1,u2,u3)
    for (int i1=1; i1 < n-4; i1+=4)
    {
      // sweep over four rows in one loop
      int i0 = 1;
      for (; i0 + simd_width < n; i0 += simd_width)
      {
        // starting indices in each row
        int index0 = i1 * n + i0;
        int index1 = (i1+1) * n + i0;
        int index2 = (i1+2) * n + i0;
        int index3 = (i1+3) * n + i0;

        // start with left neighbors in four rows
        s0.copy_from(&uold[index0 - 1], stdx::element_aligned);
        s1.copy_from(&uold[index1 - 1], stdx::element_aligned);
        s2.copy_from(&uold[index2 - 1], stdx::element_aligned);
        s3.copy_from(&uold[index3 - 1], stdx::element_aligned);
        s0 *= q;
        s1 *= q;
        s2 *= q;
        s3 *= q;

        // load 6 up and down neighbors
        t0.copy_from(&uold[index0 - n], stdx::element_aligned);
        t1.copy_from(&uold[index0], stdx::element_aligned);
        t2.copy_from(&uold[index1], stdx::element_aligned);
        t3.copy_from(&uold[index2], stdx::element_aligned);
        t4.copy_from(&uold[index3], stdx::element_aligned);
        t5.copy_from(&uold[index3 + n], stdx::element_aligned);
        s0 += q * t0; // here is the reuse!
        s1 += q * t1;
        s2 += q * t2;
        s3 += q * t3;
        s0 += q * t2;
        s1 += q * t3;
        s2 += q * t4;
        s3 += q * t5;

        // right neighbors
        t0.copy_from(&uold[index0 + 1], stdx::element_aligned);
        t1.copy_from(&uold[index1 + 1], stdx::element_aligned);
        t2.copy_from(&uold[index2 + 1], stdx::element_aligned);
        t3.copy_from(&uold[index3 + 1], stdx::element_aligned);
        s0 += q * t0;
        s1 += q * t1;
        s2 += q * t2;
        s3 += q * t3;

        // store result
        s0.copy_to(&unew[index0], stdx::element_aligned);
        s1.copy_to(&unew[index1], stdx::element_aligned);
        s2.copy_to(&unew[index2], stdx::element_aligned);
        s3.copy_to(&unew[index3], stdx::element_aligned);
      }
      // tail loop within these four rows
      for (; i0 < n - 1; i0++)
        for (int ii1=i1; ii1<i1+4; ii1++)
        {
          unew[ii1 * n + i0] = 0.25 * (uold[ii1 * n + i0 - n] + uold[ii1 * n + i0 - 1] + uold[ii1 * n + i0 + 1] + uold[ii1 * n + i0 + n]);
        }
    }
    // tail loop for rows: sequential
    for (int i1 = n-1 - ((n-2)%4); i1 < n - 1; i1++)
      for (int i0 = 1; i0 < n - 1; i0++)
        unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    std::swap(uold, unew);
  }
//  std::cout << "n=" << n << " iter=" << iterations << " countrows=" << countrows << " countcols=" << countcols << std::endl;
}

// just calls the kernel using arguments from context structure
void jacobi_omp_stdsimd2(std::shared_ptr<GlobalContext> context)
{
  jacobi_omp_stdsimd_kernel2(context->n, context->iterations, context->u0, context->u1);
}

// main function runs the experiments and outputs results as csv
int main(int argc, char **argv)
{
  int nP=omp_get_max_threads();
  std::vector<int> sizes;
  std::vector<int> iters;
  int iterations = 4096;
  for (int n = 512; n < 20000; n *= 2)
  {
    sizes.push_back(n+2);
    iters.push_back(iterations);
    iterations /= 2;
  }

  std::vector<double> performance_vanilla;
  std::vector<double> performance_stdsimd;
  std::vector<double> performance_stdsimd2;
  
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
    jacobi_omp_stdsimd(context);
    auto stop = get_time_stamp();
    double elapsed = get_duration_seconds(start, stop);
    double updates = context->iterations;
    updates *= (n - 2) * (n - 2);

    // vanilla
    std::fill(context->u0, (context->u0) + n, 0.0);
    for (int i1 = 0; i1 < n; i1++)
      for (int i0 = 0; i0 < n; i0++)
        context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
    start = get_time_stamp();
    jacobi_vanilla(context);
    stop = get_time_stamp();
    elapsed = get_duration_seconds(start, stop);
    performance_vanilla.push_back(updates / elapsed / 1e9);
    //std::cout << "vanilla n=" << n << " elapsed=" << elapsed << std::endl;

    // stdsimd1
    std::fill(context->u0, (context->u0) + n, 0.0);
    for (int i1 = 0; i1 < n; i1++)
      for (int i0 = 0; i0 < n; i0++)
        context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
    start = get_time_stamp();
    jacobi_omp_stdsimd(context);
    stop = get_time_stamp();
    elapsed = get_duration_seconds(start, stop);
    performance_stdsimd.push_back(updates / elapsed / 1e9);
    //std::cout << "simd1 n=" << n << " elapsed=" << elapsed << std::endl;

    // stdsimd2
    std::fill(context->u0, (context->u0) + n, 0.0);
    for (int i1 = 0; i1 < n; i1++)
      for (int i0 = 0; i0 < n; i0++)
        context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
    start = get_time_stamp();
    jacobi_omp_stdsimd2(context);
    stop = get_time_stamp();
    elapsed = get_duration_seconds(start, stop);
    performance_stdsimd2.push_back(updates / elapsed / 1e9);
    //std::cout << "simd2 n=" << n << " elapsed=" << elapsed << std::endl;

    // deallocate arrays
    // delete[] context->u1;
    // delete[] context->u0;
    ::operator delete[](context->u1, std::align_val_t(64));
    ::operator delete[](context->u0, std::align_val_t(64));
  }

  // print results
  std::cout << "jacobi_omp_std_simd: P=" << nP << std::endl;
  std::cout << "N, vanilla, stdsimd, stdsimd2" << std::endl;
  for (int i=0; i<sizes.size(); i++)
    std::cout << sizes[i] << ", " 
              << performance_vanilla[i] 
              << ", " << performance_stdsimd[i] 
              << ", " << performance_stdsimd2[i] 
              << std::endl;

  return 0;
}
