#include <iostream>
#include <vector>
#include <string>
#include "time_experiment.hh"

// B = A^T,
// A,B are nxn matrices stored row-major in a 1d array,
// assume A and B are NOT the same matrix
// B accessed consecutively, A accessed strided
void transpose1 (int n, double* A, double* B)
{
  for (int i=0; i<n; i++)
    for (int j=0; j<n; j++)
      B[i*n+j] = A[j*n+i];
}

// B = A^T,
// A,B are nxn matrices stored row-major in a 1d array,
// assume A and B are NOT the same matrix
// A accessed consecutively, B accessed strided
void transpose2 (int n, double* A, double* B)
{
  for (int i=0; i<n; i++)
    for (int j=0; j<n; j++)
      B[j*n+i] = A[i*n+j];
}

// B = A^T,
// A,B are nxn matrices stored row-major in a 1d array,
// assume A and B are NOT the same matrix
// B accessed consecutively, A accessed strided
template<int M, int N> // MxN blocks 
void transpose3 (int n, double* A, double* B)
{
  if (n%M!=0) {
    std::cout << n << " is not a multiple of " << M << std::endl;
    exit(0);
  }
  if (n%N!=0) {
    std::cout << n << " is not a multiple of " << N << std::endl;
    exit(0);
  }
  for (int i=0; i<n; i+=M)
    for (int j=0; j<n; j+=N)
      {
	int Bstart = i*n+j; int Astart = j*n+i;
	for (int ii=0; ii<M; ++ii)
	  for (int jj=0; jj<N; ++jj)
	    B[ii*n+jj+Bstart] = A[jj*n+ii+Astart];
      }
}

// B = A^T,
// A,B are nxn matrices stored row-major in a 1d array,
// assume A and B are NOT the same matrix
// B accessed strided, A accessed consecutively
template<int M, int N> // MxN blocks 
void transpose4 (int n, double* A, double* B)
{
  if (n%M!=0) {
    std::cout << n << " is not a multiple of " << M << std::endl;
    exit(0);
  }
  if (n%N!=0) {
    std::cout << n << " is not a multiple of " << N << std::endl;
    exit(0);
  }
  for (int i=0; i<n; i+=M)
    for (int j=0; j<n; j+=N)
      {
	int Astart = i*n+j; int Bstart = j*n+i;
	for (int ii=0; ii<M; ++ii)
	  for (int jj=0; jj<N; ++jj)
	    B[jj*n+ii+Bstart] = A[ii*n+jj+Astart];
      }
}

// B = A^T,
// A,B are nxn matrices stored row-major in a 1d array,
// assume A and B are NOT the same matrix
template<int M> // bxb blocks with M blocking in the strided direction
void transpose5 (int n, int b, double* A, double* B)
{
  if (b%M!=0) {
    std::cout << b << " is not a multiple of " << M << std::endl;
    exit(0);
  }
  if (n%b!=0) {
    std::cout << n << " is not a multiple of " << b << std::endl;
    exit(0);
  }
  // first two loops over the blocks
  for (int I=0; I<n; I+=b)
    for (int J=0; J<n; J+=b)
      for (int j=J; j<J+b; j+=M)
	for (int i=I; i<I+b; ++i)
	  for (int jj=0; jj<M; ++jj)
	    {
	      int jjj = j+jj;
	      B[i*n+jjj] = A[jjj*n+i]; // consecutive writes
	    }
}

  // B = A^T,
// A,B are nxn matrices stored row-major in a 1d array,
// assume A and B are NOT the same matrix
template<int M> // bxb blocks with M blocking in the strided direction
void transpose6 (int n, int b, double* A, double* B)
{
  if (b%M!=0) {
    std::cout << b << " is not a multiple of " << M << std::endl;
    exit(0);
  }
  if (n%b!=0) {
    std::cout << n << " is not a multiple of " << b << std::endl;
    exit(0);
  }
  // first two loops over the blocks
  for (int I=0; I<n; I+=b)
    for (int J=0; J<n; J+=b)
      for (int j=J; j<J+b; j+=M)
	for (int i=I; i<I+b; ++i)
	  for (int jj=0; jj<M; ++jj)
	    {
	      int jjj = j+jj;
	      B[jjj*n+i] = A[i*n+jjj]; // strided writes
	    }
}

// initialize square matrix
void initialize (int n, double* A)
{
  for (int i=0; i<n*n; i++) A[i] = i;
}

// conduct experiment X with transposeX
class Experiment1
{
  int n;
  double *A, *B;
public:
  // construct an experiment
  Experiment1 (int n_) : n(n_)
  {
    std::cout << "Exp1: " << n << std::endl;
    A = new double [n*n];
    B = new double [n*n];
    initialize(n,A);
    initialize(n,B);
    if (((size_t)A)%64!=0) {
      std::cout << "Exp1: A not aligned to 64 " << std::endl;
    }
    if (((size_t)B)%64!=0) {
      std::cout << "Exp1: B not aligned to 64 " << std::endl;
    }
  }
  ~Experiment1 () {delete[]A; delete[]B;}
  // run an experiment; can be called several times
  void run () const
  {
    transpose1(n,A,B);
  }
  // report number of operations for one run
  double operations () const
  {
    return n*n;
  }
};

// conduct experiment X with transposeX
class Experiment2
{
  int n;
  double *A, *B;
public:
  // construct an experiment
  Experiment2 (int n_) : n(n_)
  {
    std::cout << "Exp2: " << n << std::endl;
    A = new double [n*n];
    B = new double [n*n];
    initialize(n,A);
    initialize(n,B);
    if (((size_t)A)%64!=0) {
      std::cout << "Exp2: A not aligned to 64 " << std::endl;
    }
    if (((size_t)B)%64!=0) {
      std::cout << "Exp2: B not aligned to 64 " << std::endl;
    }
  }
  ~Experiment2 () {delete[]A; delete[]B;}
  // run an experiment; can be called several times
  void run () const
  {
    transpose2(n,A,B);
  }
  // report number of operations for one run
  double operations () const
  {
    return n*n;
  }
};

// conduct experiment X with transposeX
template<int M, int N>
class Experiment3
{
  int n;
  double *A, *B;
public:
  // construct an experiment
  Experiment3 (int n_) : n(n_)
  {
    std::cout << "Exp3: " << n << std::endl;
    A = new double [n*n];
    B = new double [n*n];
    initialize(n,A);
    initialize(n,B);
    if (((size_t)A)%64!=0) {
      std::cout << "Exp3: A not aligned to 64 " << std::endl;
    }
    if (((size_t)B)%64!=0) {
      std::cout << "Exp3: B not aligned to 64 " << std::endl;
    }
  }
  ~Experiment3 () {delete[]A; delete[]B;}
  // run an experiment; can be called several times
  void run () const
  {
    transpose3<M,N>(n,A,B);
  }
  // report number of operations for one run
  double operations () const
  {
    return n*n;
  }
};

// conduct experiment X with transposeX
template<int M, int N>
class Experiment4
{
  int n;
  double *A, *B;
public:
  // construct an experiment
  Experiment4 (int n_) : n(n_)
  {
    std::cout << "Exp4: " << n << std::endl;
    A = new double [n*n];
    B = new double [n*n];
    initialize(n,A);
    initialize(n,B);
    if (((size_t)A)%64!=0) {
      std::cout << "Exp4: A not aligned to 64 " << std::endl;
    }
    if (((size_t)B)%64!=0) {
      std::cout << "Exp4: B not aligned to 64 " << std::endl;
    }
  }
  ~Experiment4 () {delete[]A; delete[]B;}
  // run an experiment; can be called several times
  void run () const
  {
    transpose4<M,N>(n,A,B);
  }
  // report number of operations for one run
  double operations () const
  {
    return n*n;
  }
};

// conduct experiment X with transposeX
template<int M>
class Experiment5
{
  int n, b;
  double *A, *B;
public:
  // construct an experiment
  Experiment5 (int n_, int b_) : n(n_), b(b_)
  {
    std::cout << "Exp5: " << n << std::endl;
    A = new double [n*n];
    B = new double [n*n];
    initialize(n,A);
    initialize(n,B);
    if (((size_t)A)%64!=0) {
      std::cout << "Exp5: A not aligned to 64 " << std::endl;
    }
    if (((size_t)B)%64!=0) {
      std::cout << "Exp5: B not aligned to 64 " << std::endl;
    }
  }
  ~Experiment5 () {delete[]A; delete[]B;}
  // run an experiment; can be called several times
  void run () const
  {
    transpose5<M>(n,std::min(n,b),A,B);
  }
  // report number of operations for one run
  double operations () const
  {
    return n*n;
  }
};

// conduct experiment X with transposeX
template<int M>
class Experiment6
{
  int n,b;
  double *A, *B;
public:
  // construct an experiment
  Experiment6 (int n_, int b_) : n(n_), b(b_)
  {
    std::cout << "Exp6: " << n << std::endl;
    A = new double [n*n];
    B = new double [n*n];
    initialize(n,A);
    initialize(n,B);
    if (((size_t)A)%64!=0) {
      std::cout << "Exp6: A not aligned to 64 " << std::endl;
    }
    if (((size_t)B)%64!=0) {
      std::cout << "Exp6: B not aligned to 64 " << std::endl;
    }
  }
  ~Experiment6 () {delete[]A; delete[]B;}
  // run an experiment; can be called several times
  void run () const
  {
    transpose6<M>(n,std::min(n,b),A,B);
  }
  // report number of operations for one run
  double operations () const
  {
    return n*n;
  }
};

// main function runs the experiments and outputs results as csv
int main (int argc, char** argv)
{
  std::vector<int> sizes; // vector with problem sizes to try
  // for (int i=16; i<=16384; i*=2) sizes.push_back(i);
  for (int i=24; i<=25000; i*=2) sizes.push_back(i);

  std::vector<std::string> expnames; // name of experiment

  // experiment 1
  expnames.push_back("vanilla-consecutive-write");
  std::cout << expnames.back() << std::endl;
  std::vector<double> bandwidth1;
  for (auto n : sizes)
    { 
      Experiment1 e(n);
      auto d = time_experiment(e,1000000);
      double result = d.first*e.operations()*2*sizeof(double)/d.second*1e6/1e9;
      bandwidth1.push_back(result);
      std::cout << result << std::endl;
    }
  // experiment 2
  expnames.push_back("vanilla-strided-write");
  std::cout << expnames.back() << std::endl;
  std::vector<double> bandwidth2;
  for (auto n : sizes)
    { 
      Experiment2 e(n);
      auto d = time_experiment(e,1000000);
      double result = d.first*e.operations()*2*sizeof(double)/d.second*1e6/1e9;
      bandwidth2.push_back(result);
      std::cout << result << std::endl;
    }
  // // experiment 3
  // expnames.push_back("block-consecutive-write-24x4");
  // std::cout << expnames.back() << std::endl;
  // std::vector<double> bandwidth3;
  // for (auto n : sizes)
  //   { 
  //     Experiment3<24,4> e(n);
  //     auto d = time_experiment(e,1000000);
  //     double result = d.first*e.operations()*2*sizeof(double)/d.second*1e6/1e9;
  //     bandwidth3.push_back(result);
  //     std::cout << result << std::endl;
  //   }
  // // experiment 4
  // expnames.push_back("block-strided-write-4x24");
  // std::cout << expnames.back() << std::endl;
  // std::vector<double> bandwidth4;
  // for (auto n : sizes)
  //   { 
  //     Experiment4<24,4> e(n);
  //     auto d = time_experiment(e,1000000);
  //     double result = d.first*e.operations()*2*sizeof(double)/d.second*1e6/1e9;
  //     bandwidth4.push_back(result);
  //     std::cout << result << std::endl;
  //   }
  // experiment 5
  expnames.push_back("block-consecutive_write-24x4");
  std::cout << expnames.back() << std::endl;
  std::vector<double> bandwidth5;
  for (auto n : sizes)
    { 
      Experiment5<4> e(n,24);
      auto d = time_experiment(e,1000000);
      double result = d.first*e.operations()*2*sizeof(double)/d.second*1e6/1e9;
      bandwidth5.push_back(result);
      std::cout << result << std::endl;
    }
  // experiment 6
  std::cout << expnames.back() << std::endl;
  expnames.push_back("block-strided-writed-4x24");
  std::vector<double> bandwidth6;
  for (auto n : sizes)
    { 
      Experiment6<4> e(n,24);
      auto d = time_experiment(e,1000000);
      double result = d.first*e.operations()*2*sizeof(double)/d.second*1e6/1e9;
      bandwidth6.push_back(result);
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
      std::cout << ", " << bandwidth1[i];
      std::cout << ", " << bandwidth2[i];
      // std::cout << ", " << bandwidth3[i];
      // std::cout << ", " << bandwidth4[i];
      std::cout << ", " << bandwidth5[i];
      std::cout << ", " << bandwidth6[i];
      std::cout << std::endl;
    }
  
  return 0;
}
