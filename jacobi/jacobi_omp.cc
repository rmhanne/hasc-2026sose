#include "time_experiment.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

struct GlobalContext
{
  // input data
  int n;          // nxn lattice of points including boundary
  int iterations; // number of iterations to do
  double *u0;     // the initial guess
  double *u1;     // temporary vector

  GlobalContext(int n_, int iterations_) : n(n_), iterations(iterations_)
  {
    u0 = new (std::align_val_t(64)) double[n * n];
    u1 = new (std::align_val_t(64)) double[n * n];
  }

  ~GlobalContext()
  {
    ::operator delete[](u1, std::align_val_t(64));
    ::operator delete[](u0, std::align_val_t(64));
  }
};

// Sequential reference implementation of the Jacobi 5-point stencil.
// This is the correct baseline that your parallel kernels must reproduce.
void jacobi_vanilla_kernel(int n, int iterations, double *__restrict__ uold,
                           double *__restrict__ unew)
{
  for (int i = 1; i <= iterations; i++)
  {
    for (int i1 = 1; i1 < n - 1; i1++)
      for (int i0 = 1; i0 < n - 1; i0++)
        unew[i1 * n + i0] =
            0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] +
                    uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    std::swap(uold, unew);
  }
}

void jacobi_vanilla(std::shared_ptr<GlobalContext> &context)
{
  jacobi_vanilla_kernel(context->n, context->iterations, context->u0,
                        context->u1);
}

// ===========================================================================
// EXERCISE: implement your own OpenMP Jacobi kernel(s) below, following the
// same (kernel + thin wrapper) pattern as jacobi_vanilla above. Then register
// each one with verify(...) and benchmark(...) in main() to check correctness
// against the sequential reference and to measure throughput.
// ===========================================================================

int main(int argc, char **argv)
{
  // read parameters
  int n, iterations;
  if (argc != 3)
  {
    std::cout << "usage: ./jacobi_omp <size> <iterations>" << std::endl;
    exit(1);
  }
  sscanf(argv[1], "%d", &n);
  sscanf(argv[2], "%d", &iterations);
  std::cout << "jacobi: n=" << n << " iterations=" << iterations
            << " memory (mbytes)=" << (n * n) * 8.0 * 2.0 / 1024.0 / 1024.0
            << std::endl;

  const double updates = double(iterations) * (n - 2) * (n - 2);

  auto context = std::make_shared<GlobalContext>(n, iterations);

  // fill boundary values and initial values
  auto g = [&](int i0, int i1)
  {
    return (i0 > 0 && i0 < n - 1 && i1 > 0 && i1 < n - 1)
               ? 0.0
               : ((double)(i0 + i1)) / n;
  };

  auto init = [&]()
  {
    for (int i1 = 0; i1 < n; i1++)
      for (int i0 = 0; i0 < n; i0++)
        context->u0[i1 * n + i0] = context->u1[i1 * n + i0] = g(i0, i1);
  };
  init();

  // The final field ends up in u0 for an even iteration count, u1 for odd
  // (the kernels swap local pointers only, so the struct buffers don't move).
  double *result = (iterations % 2 == 0) ? context->u0 : context->u1;

  // Reference solution computed by the trusted sequential kernel.
  init();
  jacobi_vanilla(context);
  std::vector<double> reference(result, result + n * n);

  // Run a kernel from the same initial state and compare it to the reference.
  auto verify = [&](const char *name, auto kernel)
  {
    init();
    kernel(context);
    double max_diff = 0.0;
    for (int i = 0; i < n * n; i++)
      max_diff = std::max(max_diff, std::abs(result[i] - reference[i]));
    std::cout << "Verify " << name << ": max diff = " << max_diff
              << (max_diff < 1e-12 ? "  [PASS]" : "  [FAIL]") << "\n";
  };

  // Run a kernel from the same initial state and report throughput in
  // GUpdates/s (one warmup run, then a timed run).
  auto benchmark = [&](const char *name, auto kernel)
  {
    init();
    kernel(context); // warmup
    init();
    auto start = get_time_stamp();
    kernel(context);
    auto stop = get_time_stamp();
    auto elapsed = get_duration_seconds(start, stop);
    std::cout << name << ": " << updates / elapsed / 1e9 << " GUpdates/s\n";
  };

  // The reference kernel trivially passes and gives the sequential baseline.
  verify("vanilla", jacobi_vanilla);
  benchmark("vanilla", jacobi_vanilla);

  // TODO: verify and benchmark your OpenMP kernels here, e.g.
  //   verify("omp", jacobi_omp);
  //   benchmark("omp", jacobi_omp);
}
