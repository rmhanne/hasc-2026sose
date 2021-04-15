#include <iostream>
#include <vector>

#include "time_experiment.hh"

using NUMBER=double;

const int M = 16;
const int N = 256*M;
NUMBER A[N][N], B[N][N], C[N][N];

// initialize
void initialize (NUMBER A[N][N], NUMBER B[N][N], NUMBER C[N][N])
{
  int i,j,k;

  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      {
        A[i][j] = (1.0*i*j)/(N*N);
        B[i][j] = (1.0+i+j)/N;
        C[i][j] = 0.0;
      }
}

// naive matmul
void matmul1 (int n, NUMBER A[N][N], NUMBER B[N][N], NUMBER C[N][N])
{
  int i,j,k;
  NUMBER *pa,*pb,*pc;

  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      for (k=0; k<n; k++)
        C[i][j] += A[i][k]*B[k][j];
}

// naive matmul
void matmul1b (int n, NUMBER A[N][N], NUMBER B[N][N], NUMBER C[N][N])
{
  int i,j,k;
  NUMBER *pa,*pb,*pc;

  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      {
        pa = A[0]+i*N;
        pb = B[0]+j;
        pc = C[0]+i*N+j;
        for (k=0; k<n; k++)
          {
            *pc += (*pa) * (*pb);
            pa++;
            pb+=N;
          }
      }
}

// tiling
void matmul2 (int n, NUMBER A[N][N], NUMBER B[N][N], NUMBER C[N][N])
{
  int i,j,k,s,t,u;
  
  for (i=0; i<n; i+=M)
    for (j=0; j<n; j+=M)
      for (k=0; k<n; k+=M)
        for (s=0; s<M; s++)
          for (t=0; t<M; t++)
            for (u=0; u<M; u++)
              C[i+s][j+t] += A[i+s][k+u]*B[k+u][j+t];
}

// package an experiment as a functor
class Experiment1
{
  int n;
public:
  // construct an experiment
  Experiment1 (int n_) : n(n_) {initialize(A,B,C);}
  // run an experiment; can be called several times
  void run () const
  {
    matmul1(n,A,B,C);
  }
  // report number of operations
  double operations () const
  {
    return 2.0*n*n*n;
  }
};

// package an experiment as a functor
class Experiment2
{
  int n;
public:
  // construct an experiment
  Experiment2 (int n_) : n(n_) {initialize(A,B,C);}
  // run an experiment; can be called several times
  void run () const
  {
    matmul2(n,A,B,C);
  }
  // report number of operations
  double operations () const
  {
    return 2.0*n*n*n;
  }
};

int main (int argc, char** argv)
{
  std::vector<int> sizes = {M,2*M,4*M,8*M,16*M,32*M,64*M,128*M,256*M};
  std::cout << "N, vanilla, tiled" << std::endl;
  for (auto i : sizes)
    { 
      Experiment1 e1(i);
      Experiment2 e2(i);
      auto d2 = time_experiment(e2);
      auto d1 = time_experiment(e1);
      double flops1 = d1.first*e1.operations()/d1.second*1e6/1e9;
      double flops2 = d2.first*e2.operations()/d2.second*1e6/1e9;
      std::cout << i << ", " << flops1 << ", " << flops2 << std::endl;
    }
  return 0;
}
