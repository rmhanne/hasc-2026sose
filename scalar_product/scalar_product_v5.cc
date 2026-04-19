#include <iostream>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "time_experiment.hh"

using NUMBER=double;

const int P = 1;
const int N=32*1024*1024;     // problem size
std::vector<NUMBER> x(N,1.0); // first vector
std::vector<NUMBER> y(N,1.0); // second vector
std::vector<NUMBER> sums(P);  // sum of each thread
std::vector<int> flags(P,0);   // flags[i]==1 signals that thread i provides the result
std::vector<std::mutex> ms(P); // mutexes
std::vector<std::condition_variable> cvs(P); // and condition variables

// scalar product with lock-free sumation
void f (int rank, int n)
{
  // compute local sum
  NUMBER mysum=0.0;
  for (int i=(n*rank)/P; i<(n*(rank+1))/P; ++i)
    mysum += x[i]*y[i];
  sums[rank] = mysum;

  // parallel algorithm for global sum
  for (int stride=1; stride<P; stride*=2)
    if (rank%(2*stride)==0)
      {
        // add result from partner
        auto other = rank + stride;
        if (other<P)
          {
            std::unique_lock<std::mutex> lock{ms[other]};
            cvs[other].wait(lock,[other]{return flags[other]==1;});
            sums[rank] += sums[other];
            flags[other] = 0; // reset flag
          }
      }
    else
      {
        // notify that result is ready
        std::unique_lock<std::mutex> lock{ms[rank]};
        flags[rank] = 1;
        cvs[rank].notify_one();
        break;
      }
}

// package an experiment as a functor
class Experiment
{
  int n;
public:
  // construct an experiment
  Experiment (int n_) : n(n_) {}
  // run an experiment; can be called several times
  void operator() () const
  {
    for (int rank=0; rank<P; ++rank) sums[rank] = 0;
    std::vector<std::thread> threads;
    for (int rank=0; rank<P; ++rank)
      threads.push_back(std::thread{f,rank,n});
    for (int rank=0; rank<P; ++rank)
      threads[rank].join();
    if (sums[0]!=(double)n)
      std::cout << "error=" << sums[0]-n << std::endl;
  }
  // report number of operations
  double operations () const
  {
    return 2.0*n;
  }
};


int main ()
{
  std::cout << N*sizeof(NUMBER)/1024/1024 << " MByte per vector" << std::endl;
  std::cout << "using " << P << " threads" << std::endl;
  std::vector<int> sizes = {256,1024,4096,16384,65536,262144,1048576,4*1048576,16*1048576,32*1048576};
  for (auto i : sizes)
    { 
      Experiment e(i);
      auto d = time_experiment(e);
      double flops = d.first*e.operations()/d.second/1e9;
      std::cout << "n=" << i << " took " << d.second << " us for " << d.first << " repetitions"
                << " " << flops << " Gflops/s"
                << " " << flops*8 << " GByte/s"
                << std::endl;
    }
  return 0;
}
