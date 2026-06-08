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
  double *u;      // the (single) solution buffer, updated in place
  double *u1;     // scratch buffer, used by the Jacobi kernel

  GlobalContext(int n_, int iterations_) : n(n_), iterations(iterations_)
  {
    u = new (std::align_val_t(64)) double[n * n];
    u1 = new (std::align_val_t(64)) double[n * n];
  }

  ~GlobalContext()
  {
    ::operator delete[](u1, std::align_val_t(64));
    ::operator delete[](u, std::align_val_t(64));
  }
};

// Norm of the defect d^k = A u^k after k iterations, used for the convergence
// criterion. ||d^k|| <= TOL * ||d^0|| is a reasonable stopping rule.
double defect_norm(int n, double *__restrict__ u)
{
  double sum = 0.0;
  for (int i1 = 1; i1 < n - 1; i1++)
    for (int i0 = 1; i0 < n - 1; i0++)
    {
      double d = 4.0 * u[i1 * n + i0] -
                 (u[i1 * n + i0 - n] + u[i1 * n + i0 - 1] +
                  u[i1 * n + i0 + 1] + u[i1 * n + i0 + n]);
      sum += d * d;
    }
  return std::sqrt(sum);
}

// Sequential reference implementation of the Gauss-Seidel 5-point stencil.
void gauss_seidel_vanilla_kernel(int n, int iterations, double *__restrict__ u)
{
  for (int i = 1; i <= iterations; i++)
    for (int i1 = 1; i1 < n - 1; i1++)
      for (int i0 = 1; i0 < n - 1; i0++)
        u[i1 * n + i0] = 0.25 * (u[i1 * n + i0 - n] + u[i1 * n + i0 - 1] +
                                 u[i1 * n + i0 + 1] + u[i1 * n + i0 + n]);
}

void gauss_seidel_vanilla(std::shared_ptr<GlobalContext> &context)
{
  gauss_seidel_vanilla_kernel(context->n, context->iterations, context->u);
}

// Sequential Jacobi 5-point stencil
void jacobi_vanilla_kernel(int n, int iterations, double *__restrict__ u,
                           double *__restrict__ tmp)
{
  std::copy(u, u + n * n, tmp); // copy boundary (and initial interior) values

  double *uold = u;
  double *unew = tmp;
  for (int i = 1; i <= iterations; i++)
  {
    for (int i1 = 1; i1 < n - 1; i1++)
      for (int i0 = 1; i0 < n - 1; i0++)
        unew[i1 * n + i0] =
            0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 - 1] +
                    uold[i1 * n + i0 + 1] + uold[i1 * n + i0 + n]);
    std::swap(uold, unew);
  }

  if (uold != u) // final result landed in tmp -> copy it back into u
    std::copy(uold, uold + n * n, u);
}

void jacobi_vanilla(std::shared_ptr<GlobalContext> &context)
{
  jacobi_vanilla_kernel(context->n, context->iterations, context->u,
                        context->u1);
}

// Run Gauss-Seidel in place until the defect norm has been reduced by a factor
// tol relative to the initial defect. Since the norm is expensive, it is only
// evaluated every check_interval sweeps. Returns the number of sweeps done.
int gauss_seidel_solve(int n, double *__restrict__ u, double tol,
                       int check_interval)
{
  // TODO
}

// Same convergence loop for Jacobi (two buffers, only old neighbour values).
// Uses the caller-provided scratch buffer tmp (the context's u1) as the second
// buffer; the final iterate is written back into u. Returns the number of sweeps done.
int jacobi_solve(int n, double *__restrict__ u, double *__restrict__ tmp,
                 double tol, int check_interval)
{
  // TODO
}

// Run both methods from the same initial state until the defect has dropped by
// a factor tol, and report how many sweeps and how much time each one needed.
// This is the fair way to compare them: Gauss-Seidel typically needs about half
// as many sweeps as Jacobi, but each sweep is harder to parallelise.
template <typename Init>
void compare_convergence(std::shared_ptr<GlobalContext> &context, double tol,
                         int check_interval, Init init)
{
  const int n = context->n;
  double *u = context->u;

  init();
  auto t0 = get_time_stamp();
  int gs_iters = gauss_seidel_solve(n, u, tol, check_interval);
  auto t1 = get_time_stamp();

  init();
  auto t2 = get_time_stamp();
  int ja_iters = jacobi_solve(n, u, context->u1, tol, check_interval);
  auto t3 = get_time_stamp();

  std::cout << "convergence to tol=" << tol << " (defect checked every "
            << check_interval << " sweeps):\n";
  std::cout << "  gauss-seidel: " << gs_iters << " sweeps, "
            << get_duration_seconds(t0, t1) << " s\n";
  std::cout << "  jacobi:       " << ja_iters << " sweeps, "
            << get_duration_seconds(t2, t3) << " s\n";
  std::cout << "  sweep ratio jacobi/gauss-seidel = "
            << double(ja_iters) / gs_iters << "\n";
}

// ===========================================================================
// EXERCISE: implement your own Gauss-Seidel kernel(s) below, following the
// same (kernel + thin wrapper) pattern as gauss_seidel_vanilla above. Then
// register each one with verify(...) and benchmark(...) in main() to check
// correctness against the sequential reference and to measure throughput.
// ===========================================================================

int main(int argc, char **argv)
{
  // read parameters
  int n, iterations;
  double tol = 1e-3;       // stop when the defect has dropped by this factor
  int check_interval = 10; // evaluate the (expensive) defect norm this often
  if (argc < 3 || argc > 5)
  {
    std::cout << "usage: ./gauss_seidel_omp <size> <iterations> [tol] "
                 "[check_interval]"
              << std::endl;
    exit(1);
  }
  sscanf(argv[1], "%d", &n);
  sscanf(argv[2], "%d", &iterations);
  if (argc >= 4)
    sscanf(argv[3], "%lf", &tol);
  if (argc >= 5)
    sscanf(argv[4], "%d", &check_interval);
  std::cout << "gauss-seidel: n=" << n << " iterations=" << iterations
            << " memory (mbytes)=" << (n * n) * 8.0 * 2.0 / 1024.0 / 1024.0
            << std::endl;

  const double updates = double(iterations) * (n - 2) * (n - 2);

  auto context = std::make_shared<GlobalContext>(n, iterations);
  double *result = context->u;

  // fill boundary values and interior initial values
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
        context->u[i1 * n + i0] = g(i0, i1);
  };

  // Reference solution computed by the trusted sequential kernel.
  init();
  gauss_seidel_vanilla(context);
  std::vector<double> reference(result, result + n * n);

  // Run a kernel from the same initial state and compare it to the reference.
  // Because the pipeline is only a different traversal order of the SAME
  // dependency graph, a correct parallel kernel reproduces the reference
  // exactly (max diff == 0). Red-black ordering is a DIFFERENT iterate and is
  // not expected to pass this test.
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
  verify("vanilla", gauss_seidel_vanilla);
  benchmark("vanilla", gauss_seidel_vanilla);

  // Jacobi throughput for comparison (it is a different iterate, so it is not
  // verified against the Gauss-Seidel reference).
  benchmark("jacobi", jacobi_vanilla);

  // Compare how many sweeps each method needs to reach the tolerance.
  compare_convergence(context, tol, check_interval, init);

  // TODO: verify and benchmark your kernels here, e.g.
  //   verify("wavefront", gauss_seidel_wavefront);
  //   benchmark("wavefront", gauss_seidel_wavefront);
}
