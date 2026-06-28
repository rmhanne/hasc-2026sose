#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <cassert>

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
  int tag = 50;              // from slide 9 of lecture 20 example
  int n = context->n;
  int nloc = context->nloc;
  int up = context->up;
  int down = context->down;
  int rank = context->rank;
  MPI_Status status;

  if (rank % 2 == 0)
  {
    MPI_Send(&u[1 * n + 0], n, MPI_DOUBLE, up, tag, comm);
    MPI_Recv(&u[0 * n + 0], n, MPI_DOUBLE, up, tag, comm, &status);

    MPI_Send(&u[nloc * n + 0], n, MPI_DOUBLE, down, tag, comm);
    MPI_Recv(&u[(nloc + 1) * n + 0], n, MPI_DOUBLE, down, tag, comm, &status);
  }
  else
  {
    MPI_Recv(&u[(nloc + 1) * n + 0], n, MPI_DOUBLE, down, tag, comm, &status);
    MPI_Send(&u[nloc * n + 0], n, MPI_DOUBLE, down, tag, comm);

    MPI_Recv(&u[0 * n + 0], n, MPI_DOUBLE, up, tag, comm, &status);
    MPI_Send(&u[1 * n + 0], n, MPI_DOUBLE, up, tag, comm);
  }
  // added this to check, which block sends to which other blocks
  //std::cout << "rank " << rank << " up=" << context->up << " down=" << context->down << std::endl;
}

void halo_exchange_nb(MPI_Comm comm, std::shared_ptr<GlobalContext> context, double *__restrict__ u, MPI_Request req[4])
{
  int n = context->n;
  int nloc = context->nloc;
  int up = context->up;
  int down = context->down;
  int tag = 50;

  // Post receives first
  MPI_Irecv(&u[0 * n],        n, MPI_DOUBLE, up,   tag, comm, &req[0]);
  MPI_Irecv(&u[(nloc + 1)*n], n, MPI_DOUBLE, down, tag, comm, &req[1]);

  // Then sends
  MPI_Isend(&u[1 * n],        n, MPI_DOUBLE, up,   tag, comm, &req[2]);
  MPI_Isend(&u[nloc * n],     n, MPI_DOUBLE, down, tag, comm, &req[3]);
  //std::cout << "rank " << context->rank << " up=" << context->up << " down=" << context->down << std::endl;
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
void jacobi_kernel_blocking(MPI_Comm comm, std::shared_ptr<GlobalContext> context) // changed the naming to keep this older version in the code :) -Marvin
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

void jacobi_kernel(MPI_Comm comm, std::shared_ptr<GlobalContext> context) {
  const int n = context->n;
  const int nloc = context->nloc;
  double *uold = context->u0;
  double *unew = context->u1;

  MPI_Request req[4];

  // the following numbers refer to the numbers of the task on the sheet :)
  for (int it = 0; it < context->iterations; ++it) {
    // (i) send boundary data
    halo_exchange_nb(comm, context, uold, req);

    // (ii) compute interior rows
    for (int i1 = 2; i1 <= nloc - 1; ++i1) {
      for (int i0 = 1; i0 < n - 1; ++i0) {
        unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 + n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1]); 
      }
    }

    // (iii) wait for halo exchange to complete
    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);

    // (iv) update boundary rows
    for (int i0 = 1; i0 < n - 1; ++i0) {
      int i1 = 1;
      unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 + n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1]);

      i1 = nloc;
      unew[i1 * n + i0] = 0.25 * (uold[i1 * n + i0 - n] + uold[i1 * n + i0 + n] + uold[i1 * n + i0 - 1] + uold[i1 * n + i0 + 1]);
    }

    std::swap(uold, unew);
  }

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

  assert(n % size == 0);

  int nloc = n / size;

  context->nloc = nloc;
  context->row_offset = rank * nloc;
  //if (rank < size - 1)  context->up = rank + 1;         // Not sure but should up not be rank-1 ?
  //if (rank > 0)  context->down = rank - 1;
  context->up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
  context->down = (rank < size-1) ? rank + 1 : MPI_PROC_NULL;
  context->u0 = new double[(nloc + 2) * n];
  context->u1 = new double[(nloc + 2) * n];

  // unchanged from jacobi_seq.cc
  //
    // fill boundary values and initial values
    auto g = [&](int i0, int i1)
    { return (i0 > 0 && i0 < n - 1 && i1 > 0 && i1 < n - 1)
                 ? 0.0
                 : ((double)(i0 + i1)) / n; };

  // modified from jacobi_seq.cc
    // warmup
    for (int i1 = 0; i1 < nloc; i1++)
      for (int i0 = 0; i0 < n; i0++)   // (1 + i1) because row 0 starts at 1
        context->u0[(1 + i1) * n + i0] = context->u1[(1 + i1) * n + i0]
                                       = g(i0, context->row_offset + i1);
 
  jacobi_kernel_blocking(MPI_COMM_WORLD, context);

  MPI_Finalize();
}
