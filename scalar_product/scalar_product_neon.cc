#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <arm_neon.h>

#include "time_experiment.hh"

using NUMBER=double;

const std::size_t P = 1; // std::thread::hardware_concurrency();
const std::size_t N=128*1024*1024; // problem size
std::vector<NUMBER> x(N,1.0); // first vector
std::vector<NUMBER> y(N,1.0); // second vector
NUMBER sum=0.0;    // result
std::mutex m;      // ensure exclusive access to sum

// loop unrolled and vectorizedscalarproduct
void f (std::size_t rank, std::size_t n) // for simplicity arguments and result are global
{
  float64x2_t sum0, sum1, sum2, sum3;
  float64x2_t x0, x1, x2, x3;
  float64x2_t y0, y1, y2, y3;
  sum0 = vmovq_n_f64(0.0);
  sum1 = vmovq_n_f64(0.0);
  sum2 = vmovq_n_f64(0.0);
  sum3 = vmovq_n_f64(0.0);
  for (std::size_t i=(n*rank)/P; i<(n*(rank+1))/P; i+=8)
  {
    x0 = vld1q_f64(&(x[i]));
    x1 = vld1q_f64(&(x[i+2]));
    x2 = vld1q_f64(&(x[i+4]));
    x3 = vld1q_f64(&(x[i+6]));
    y0 = vld1q_f64(&(y[i]));
    y1 = vld1q_f64(&(y[i+2]));
    y2 = vld1q_f64(&(y[i+4]));
    y3 = vld1q_f64(&(y[i+6]));
    sum0 = vfmaq_f64(sum0, x0, y0); // fused multiply add
    sum1 = vfmaq_f64(sum1, x1, y1); // fused multiply add
    sum2 = vfmaq_f64(sum2, x2, y2); // fused multiply add
    sum3 = vfmaq_f64(sum3, x3, y3); // fused multiply add
  }
  // horizontal add
  double s0 = vdupd_laneq_f64(sum0,0);
  double s1 = vdupd_laneq_f64(sum0,1);
  double s2 = vdupd_laneq_f64(sum1,0);
  double s3 = vdupd_laneq_f64(sum1,1);
  double s4 = vdupd_laneq_f64(sum2,0);
  double s5 = vdupd_laneq_f64(sum2,1);
  double s6 = vdupd_laneq_f64(sum3,0);
  double s7 = vdupd_laneq_f64(sum3,1);
  double mysum = s0+s1+s2+s3+s4+s5+s6+s7;
  std::lock_guard<std::mutex> lock{m};
  sum += mysum;
}

// package an experiment as a functor
class Experiment_par {
  std::size_t n;
public:
  // construct an experiment
  Experiment_par (std::size_t n_) : n(n_) {}
  // run an experiment; can be called several times
  void operator() () const
  {
    sum = 0.0;
    std::vector<std::thread> threads;
    for (std::size_t rank=0; rank<P; ++rank)
      threads.push_back(std::thread{f,rank,n});
    for (std::size_t rank=0; rank<P; ++rank)
      threads[rank].join();
  }
  // report number of operations
  double operations () const {return 2.0*n;}
};

class Experiment_seq {
  std::size_t n;
public:
  // construct an experiment
  Experiment_seq (std::size_t n_) : n(n_) {}
  // run an experiment; can be called several times
  void operator() () const
  {
    sum = 0.0;
    f(1,n);
  }
  // report number of operations
  double operations () const {return 2.0*n;}
};

int main ()
{
  std::cout << N*sizeof(NUMBER)/1024/1024 << " MByte per vector" << std::endl;
  std::cout << "using " << P << " threads" << std::endl;
  std::vector<std::size_t> sizes = {64,128,256,1024,4096,16384,65536,262144,1048576,4*1048576,16*1048576,32*1048576,64*1048576};
  for (auto i : sizes) { 
    Experiment_par e(i);
    auto d = time_experiment(e);
    double flops = d.first*e.operations()/d.second/1e9;
    std::cout << "n=" << i << " took " << d.second << " us for " << d.first << " repetitions"
	      << " " << flops << " Gflops/s"
	      << " " << flops*sizeof(NUMBER) << " GByte/s" << std::endl;
  }
  return 0;
}
