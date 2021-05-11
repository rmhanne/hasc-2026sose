#include <iostream>
#include <vector>
#include <string>
#include "time_experiment.hh"

#include <Eigen/Dense>

#include "mkl.h"

// initialize square matrix
void initialize (int n, double *a){
  for (int i=0; i<n*n; i++) a[i] = i;
}
template <typename T>
void initialize_eigen(T& a){
  typename T::value_type value(0);
  for (int i=0; i<a.rows(); ++i)
    for (int j=0; j<a.cols(); ++j){
      a(i,j) = value;
      value += 1;
    }
}

void transpose_mkl(const int n, const double *a, double *b){
    mkl_domatcopy('r', 't', n, n, 1.0, a, n, b, n);
}

class ExperimentMkl
{
  mutable double *_a, *_b;
  int _n;
public:
  ExperimentMkl (int n) : _n(n){
    // std::cout << "Exp mkl: " << _n << std::endl;
    _a = (double *) mkl_malloc(_n*_n*sizeof(double), 64);
    _b = (double *) mkl_malloc(_n*_n*sizeof(double), 64);
    initialize(n, _a);
    initialize(n, _b);
  }
  ~ExperimentMkl (){
    mkl_free(_a);
    mkl_free(_b);
  }
  void run () const {
    transpose_mkl(_n, _a, _b);
  }
  double operations () const {
    return _n * _n;
  }
};

template<typename T>
void transpose_eigen(const T& a, T& b){
  b = a;
  b.transposeInPlace();
}

class ExperimentEigen
{
  const int _n;
  mutable Eigen::MatrixXd _a, _b;
public:
  ExperimentEigen(int n) : _n(n), _a(n,n), _b(n,n){
    // std::cout << "Exp eigen: " << _n << std::endl;
    initialize_eigen(_a);
  }
  void run() const{
    transpose_eigen(_a, _b);
  }
  double operations () const {
    return _n * _n;
  }
};


void print_matrix(int n, double *a){
  for (int i=0; i<n; ++i){
    for (int j=0; j<n; ++j)
      std::cout << a[i*n+j] << " ";
    std::cout << std::endl;
  }
}
void test_matrix_transpose(){
  const int n = 24;
  double *a, *b;
  a = (double *) mkl_malloc(n*n*sizeof(double), 64);
  b = (double *) mkl_malloc(n*n*sizeof(double), 64);
  initialize(n, a);
  print_matrix(n, a);
  std::cout << std::endl;
  transpose_mkl(n, a, b);
  print_matrix(n, b);
  mkl_free(a);
  mkl_free(b);

  std::cout << std::endl;
  Eigen::MatrixXd a2(n, n), b2(n, n);
  initialize_eigen(a2);
  transpose_eigen(a2, b2);
  std::cout << b2 << std::endl;
}


// main function runs the experiments and outputs results as csv
int main (int argc, char** argv)
{
  // test_matrix_transpose();

  std::vector<int> sizes; // vector with problem sizes to try
  // for (int i=16; i<=16384; i*=2) sizes.push_back(i);
  for (int i=24; i<=16384; i*=2) sizes.push_back(i);
  // for (int i=24; i<=25000; i*=2) sizes.push_back(i);

  std::vector<std::string> expnames; // name of experiment

  expnames.push_back("intel-mkl");
  std::cout << expnames.back() << std::endl;
  std::vector<double> bandwidth1;
  for (auto n : sizes){
    ExperimentMkl e(n);
    auto d = time_experiment(e,1000000);
    double result = d.first*e.operations()*2*sizeof(double)/d.second*1e6/1e9;
    bandwidth1.push_back(result);
    std::cout << result << std::endl;
  }

  expnames.push_back("eigen");
  std::cout << expnames.back() << std::endl;
  std::cout << "Note: Eigen numbers not fair as eigen needs to copy and transpose seperately." << std::endl;
  std::vector<double> bandwidth2;
  for (auto n : sizes){
    ExperimentEigen e(n);
    auto d = time_experiment(e,1000000);
    double result = d.first*e.operations()*2*sizeof(double)/d.second*1e6/1e9;
    bandwidth2.push_back(result);
    std::cout << result << std::endl;
  }

  // output results
  // Note: size of TLB mentioned in https://www.realworldtech.com/haswell-cpu/5/
  std::cout << "N";
  for (std::string s : expnames)
    std::cout << ", " << s;
  std::cout << std::endl;
  for (std::size_t i=0; i<sizes.size(); i++){
    std::cout << sizes[i];
    std::cout << ", " << bandwidth1[i];
    std::cout << ", " << bandwidth2[i];
    std::cout << std::endl;
  }

  return 0;
}
