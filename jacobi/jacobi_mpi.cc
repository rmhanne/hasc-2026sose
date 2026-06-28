#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>

#include <mpi.h>

struct GlobalContext
{
  // input data
  int n;          // nxn global lattice of points including boundary
  int iterations; // number of iterations to do

  // distributed decomposition (1D, contiguous row strips)
  int rank = 0;          // this rank
  int size = 1;          // number of ranks
  int nloc = 0;          // number of interior rows owned by this rank
  int row_offset = 0;    // global index of this rank's first interior row
  int up = MPI_PROC_NULL;   // neighbour rank above (smaller row indices)
  int down = MPI_PROC_NULL; // neighbour rank below (larger row indices)

  // local buffers, each of size (nloc + 2) * n:
  //   row 0        : top ghost row
  //   rows 1..nloc : owned interior rows
  //   row nloc + 1 : bottom ghost row
  double *u0 = nullptr; // the initial guess
  double *u1 = nullptr; // temporary vector

  // output data

  GlobalContext(int n_)
      : n(n_)
  {
  }
  GlobalContext(int n_, int iterations_)
      : n(n_), iterations(iterations_)
  {
  }
  GlobalContext(int n_, int iterations_, int rank_, int size_)
      : n(n_), iterations(iterations_), rank(rank_), size(size_)
  {
  }
};

// Exchange ghost rows of buffer u with the up/down neighbours.
//
// TODO: Send your topmost interior row (row 1) to context->up and receive the
// neighbour's border row into the top ghost row (row 0); send your bottommost
// interior row (row context->nloc) to context->down and receive into the bottom
// ghost row (row context->nloc + 1). context->up / context->down are set to
// MPI_PROC_NULL at the domain ends, so those sends/receives become no-ops.
// Use a deadlock-free scheme (MPI_Sendrecv, even/odd ordering, or Isend/Irecv).
void halo_exchange(MPI_Comm comm, std::shared_ptr<GlobalContext> context, double *__restrict__ u)
{
}

// compute norm of defect on the parallel communicator comm
double defect_norm(MPI_Comm comm, int n, double *__restrict__ u)
{

  // @Marvin & @Hermann - I am not sure if we are allowed to change the signature of the method. But assuming we aren't allowed
  // to, I recalculate the rank size here. Otherwise, I think simply passing the context would be smarter as they already contain the computed values
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int _rows = n - 2; // Internal rows
  int _nloc = _rows / size;
  int _rem  = _rows % size;

  if ( rank < _rem ) _nloc ++;
  double lsum = 0.0;
  double gsum = 0.0;

  for (int i1=0;i1<=_nloc;++i1) {
    for (int i0=1;i0<n-1;++i0) {
      double def = 4.0 * u[i1*n + i0] -
          (u[(i1-1) * n + i0] +
           u[(i1+1) * n + i0] +
           u[(i1) * n + i0-1] +
           u[(i1) * n + i0+1]);

      lsum += def * def;
    }
  }

  MPI_Allreduce(&lsum, &gsum, 1, MPI_DOUBLE, MPI_SUM, comm);

  return std::sqrt(gsum);
}

// One Jacobi sweep over the local strip, repeated context->iterations times.
// Before each iteration the ghost rows of the source buffer are refreshed via
// halo_exchange, then every owned interior row (1..nloc) is updated.
void jacobi_kernel(MPI_Comm comm, std::shared_ptr<GlobalContext> context)
{
  const int n = context->n;
  const int nloc = context->nloc;
  double *uold = context->u0;
  double *unew = context->u1;

  for (int it = 0; it < context->iterations; ++it)
  {
    // bring the ghost rows of the current iterate up to date
    halo_exchange(comm, context, uold);

    // update all owned interior rows (1..nloc) and interior columns (1..n-2)
    for (int i1 = 1; i1 <= nloc; ++i1)
      for (int i0 = 1; i0 < n - 1; ++i0)
        unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] +
                                    uold[i1 * n + i0 - 1] +
                                    uold[i1 * n + i0 + 1] +
                                    uold[i1 * n + i0 + n]);

    std::swap(uold, unew);
  }

  // make the final iterate available as u0 again
  context->u0 = uold;
  context->u1 = unew;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  std::cout << "Rank " << rank << ": Running parallel Jacobi program with " << size << " ranks\n";

  int n = (argc > 1) ? std::atoi(argv[1]) : 512;
  int iterations = (argc > 2) ? std::atoi(argv[2]) : 1000;

  auto context = std::make_shared<GlobalContext>(n, iterations, rank, size);

  // TODO (b): determine the local decomposition (nloc, row_offset, up, down),
  // allocate context->u0 / context->u1 of size (nloc + 2) * n, and initialize
  // the strip and boundary/ghost values. Then run jacobi_kernel(MPI_COMM_WORLD,
  // context) and verify against the sequential reference.

  MPI_Finalize();
}
