#include <iostream>
#include <vector>
#include <string>
#include "time_experiment.hh"
#include "../vcl/vectorclass.h"

// y = Ax, matrix assumed to be column major now
// use SIMD instructions
template<int M>
void matvec4(int n, const double* A, const double* x, double* y)
{
  if (n%M!=0) {
    std::cout << n << " is not a multiple of " << M << std::endl;
    exit(0);
  }
  if (M%4!=0) {
    std::cout << M << " is not a multiple of " << 4 << std::endl;
    exit(0);
  }
  if (M/4>12) {
    std::cout << " too many registers " << M/4 << std::endl;
    exit(0);
  }
  Vec4d X, Y[M/4], Acol;
  for (int i=0; i<n; i++)
    y[i] = 0.0;
  for (int I=0; I<n; I+=M)
    {
      for (int i=0; i<M/4; ++i)
	Y[i].load(&y[I+4*i]);
      for (int j=0; j<n; ++j)
	{
	  X = Vec4d(x[j]); // load broadcast
	  // access to A is still consecutive and x[j] is reused
	  int k = j*n+I;
	  for (int i=0; i<M/4; ++i)
	    {
	      Acol.load(&A[k+4*i]); // load four values in column of A
	      Y[i] = mul_add(Acol,X,Y[i]);
	    }
	}
      for (int i=0; i<M/4; ++i)
	Y[i].store(&y[I+4*i]);
    }
}

// initialize an array
void initialize (int n, double* A)
{
  for (int i=0; i<n; i++) A[i] = i;
}

template<int M>
class Experiment4
{
  int n;
  double *A, *x, *y;
public:
  // construct an experiment
  Experiment4 (int n_) : n(n_)
  {
    std::cout << "Exp4: " << n << std::endl;
    A = new double [n*n];
    if (alignment(A)<=32)
      std::cout << "Exp4: A aligned to " << alignment(A) << std::endl;
    initialize(n*n,A);
    x = new double [n];
    if (alignment(x)<=32)
      std::cout << "Exp4: x aligned to " << alignment(x) << std::endl;
    initialize(n,x);
    y = new double [n];
    if (alignment(y)<=32)
      std::cout << "Exp4: y aligned to " << alignment(y) << std::endl;
    initialize(n,y);
  }
  size_t alignment (const double *p)
  {
    for (size_t m=64; m>1; m/=2)
      if (((size_t)p)%m==0) return m;
    return 1;
  }
  ~Experiment4 () {delete[]y; delete[]x;  delete[]A;}
  // run an experiment; can be called several times
  void operator() () const
  {
    matvec4<M>(n,A,x,y);
  }
  // report number of operations for one run
  double operations () const
  {
    return 2*n*n;
  }
};


// main function runs the experiments and outputs results as csv
int main (int argc, char** argv)
{
  std::vector<int> sizes; // vector with problem sizes to try
  for (int i=32; i<=25000; i*=2) sizes.push_back(i);

  std::vector<std::string> expnames; // name of experiment

  // experiment 4
  expnames.push_back("blocked-column-major-AVX-32x32");
  std::cout << expnames.back() << std::endl;
  std::vector<double> bandwidth4;
  for (auto n : sizes)
    { 
      Experiment4<32> e(n);
      auto d = time_experiment(e,1000000);
      double result = d.first*e.operations()/d.second/1e9;
      bandwidth4.push_back(result);
      std::cout << result << std::endl;
    }

  // output results
  // Note: size of TLB mentioned in https://www.realworldtech.com/haswell-cpu/5/
  std::cout << "N";
  for (std::string s : expnames)
    std::cout << ", " << s;
  std::cout << std::endl;
  for (int i=0; i<sizes.size(); i++)
    {
      std::cout << sizes[i];
      std::cout << ", " << bandwidth4[i];
      std::cout << std::endl;
    }
  
  return 0;
}
