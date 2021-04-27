#include <iostream>
#include <vector>
#include <string>
#include "time_experiment.hh"

// y = Ax, matrix assumed to be row major
void matvec1 (int n, const double* A, const double* x, double* y)
{
  for (int i=0; i<n; i++)
    {
      y[i] = 0.0;
      for (int j=0; j<n; j++) // inner loop is a scalar product
	y[i] += A[i*n+j]*x[j];
    }
}

// y = Ax, matrix assumed to be column major now
void matvec2 (int n, const double* A, const double* x, double* y)
{
  for (int i=0; i<n; i++) y[i] = 0.0;
  for (int j=0; j<n; j++) {
    for (int i=0; i<n; ++i)
      y[i] += A[j*n+i]*x[j];
  }
}

// y = Ax, matrix assumed to be column major now
void matvec3 (int n, int b, const double* A, const double* x, double* y)
{
  if (n%b!=0) {
    std::cout << n << " is not a multiple of " << b << std::endl;
    exit(0);
  }
  if (b%4!=0) {
    std::cout << b << " is not a multiple of " << 4 << std::endl;
    exit(0);
  }
  for (int i=0; i<n; i++)
    y[i] = 0.0;
  for (int I=0; I<n; I+=b)
    for (int J=0; J<n; J+=b)
      for (int j=J; j<J+b; ++j)
	for (int i=I; i<I+b; ++i)
	  y[i] += A[j*n+i]*x[j];
}

// initialize an array
void initialize (int n, double* A)
{
  for (int i=0; i<n; i++) A[i] = i;
}

class Experiment1
{
  int n;
  double *A, *x, *y;
public:
  // construct an experiment
  Experiment1 (int n_) : n(n_)
  {
    std::cout << "Exp1: " << n << std::endl;
    A = new double [n*n];
    if (((size_t)A)%64!=0)
      std::cout << "Exp1: A not aligned to 64 " << std::endl;
    initialize(n*n,A);
    x = new double [n];
    if (((size_t)x)%64!=0)
      std::cout << "Exp1: x not aligned to 64 " << std::endl;
    initialize(n,x);
    y = new double [n];
    if (((size_t)y)%64!=0)
      std::cout << "Exp1: y not aligned to 64 " << std::endl;
    initialize(n,y);
  }
  ~Experiment1 () {delete[]y; delete[]x;  delete[]A;}
  // run an experiment; can be called several times
  void run () const
  {
    matvec1(n,A,x,y);
  }
  // report number of operations for one run
  double operations () const
  {
    return 2*n*n;
  }
};

class Experiment2
{
  int n;
  double *A, *x, *y;
public:
  // construct an experiment
  Experiment2 (int n_) : n(n_)
  {
    std::cout << "Exp2: " << n << std::endl;
    A = new double [n*n];
    if (((size_t)A)%64!=0)
      std::cout << "Exp2: A not aligned to 64 " << std::endl;
    initialize(n*n,A);
    x = new double [n];
    if (((size_t)x)%64!=0)
      std::cout << "Exp2: x not aligned to 64 " << std::endl;
    initialize(n,x);
    y = new double [n];
    if (((size_t)y)%64!=0)
      std::cout << "Exp2: y not aligned to 64 " << std::endl;
    initialize(n,y);
  }
  ~Experiment2 () {delete[]y; delete[]x;  delete[]A;}
  // run an experiment; can be called several times
  void run () const
  {
    matvec2(n,A,x,y);
  }
  // report number of operations for one run
  double operations () const
  {
    return 2*n*n;
  }
};

class Experiment3
{
  int n,b;
  double *A, *x, *y;
public:
  // construct an experiment
  Experiment3 (int n_, int b_) : n(n_), b(b_)
  {
    std::cout << "Exp3: " << n << std::endl;
    A = new double [n*n];
    if (((size_t)A)%64!=0)
      std::cout << "Exp3: A not aligned to 64 " << std::endl;
    initialize(n*n,A);
    x = new double [n];
    if (((size_t)x)%64!=0)
      std::cout << "Exp3: x not aligned to 64 " << std::endl;
    initialize(n,x);
    y = new double [n];
    if (((size_t)y)%64!=0)
      std::cout << "Exp3: y not aligned to 64 " << std::endl;
    initialize(n,y);
  }
  ~Experiment3 () {delete[]y; delete[]x;  delete[]A;}
  // run an experiment; can be called several times
  void run () const
  {
    matvec3(n,std::min(n,b),A,x,y);
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

  // experiment 1
  expnames.push_back("vanilla-row-major");
  std::cout << expnames.back() << std::endl;
  std::vector<double> bandwidth1;
  for (auto n : sizes)
    { 
      Experiment1 e(n);
      auto d = time_experiment(e,1000000);
      double result = d.first*e.operations()/d.second*1e6/1e9;
      bandwidth1.push_back(result);
      std::cout << result << std::endl;
    }
  // experiment 2
  expnames.push_back("vanilla-column-major");
  std::cout << expnames.back() << std::endl;
  std::vector<double> bandwidth2;
  for (auto n : sizes)
    { 
      Experiment2 e(n);
      auto d = time_experiment(e,1000000);
      double result = d.first*e.operations()/d.second*1e6/1e9;
      bandwidth2.push_back(result);
      std::cout << result << std::endl;
    }
  // experiment 3
  expnames.push_back("blocked-column-major-64x64");
  std::cout << expnames.back() << std::endl;
  std::vector<double> bandwidth3;
  for (auto n : sizes)
    { 
      Experiment3 e(n,64);
      auto d = time_experiment(e,1000000);
      double result = d.first*e.operations()/d.second*1e6/1e9;
      bandwidth3.push_back(result);
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
      std::cout << ", " << bandwidth3[i];
      std::cout << std::endl;
    }
  
  return 0;
}
