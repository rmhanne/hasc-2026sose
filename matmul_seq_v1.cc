#include <iostream>
#include <vector>

#include "time_experiment.hh"

const int M = 64;
const int N = 64*M;
double A[N][N] __attribute__((aligned(32)));
double B[N][N] __attribute__((aligned(32)));
double C[N][N] __attribute__((aligned(32)));

// initialize
void initialize (double A[N][N], double B[N][N], double C[N][N])
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

// naive matmul C = A*B + C
void matmul1 (int n, double A[N][N], double B[N][N], double C[N][N])
{
  int i,j,k;

  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      for (k=0; k<n; k++)
        C[i][j] += A[i][k]*B[k][j];
}

// same with tiling  C = A*B + C
void matmul2 (int n, double A[N][N], double B[N][N], double C[N][N])
{
  int i,j,k,s,t,u;
  
  for (i=0; i<n; i+=M)
    for (j=0; j<n; j+=M)
      for (k=0; k<n; k+=M)
        for (s=i; s<i+M; s++)
	  for (t=j; t<j+M; t++)
	    for (u=k; u<k+M; u++)
              C[s][t] += A[s][u]*B[u][t]; 
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
  std::vector<int> sizes;
  for (int i=M; i<=5000; i*=2) sizes.push_back(i);
  std::cout << "N, vanillaautovec, tiledautovec" << std::endl;
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
