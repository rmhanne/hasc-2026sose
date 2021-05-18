#ifndef HASC_BARRIER_HH
#define HASC_BARRIER_HH

#include <vector>
#include <mutex>
#include <condition_variable>

// barrier as a class
class Barrier
{
  int P; // number of threads in barrier
  int count; // count number of threads that arrived at the barrier
  std::vector<int> flag; // flag indicating waiting thread
  std::mutex mx; // mutex for use with the cvs
  std::vector<std::condition_variable> cv; // for waiting
public:
  // set up barrier for given number of threads
  Barrier (int P_) : P(P_), count(0), flag(P_,0), cv(P_)
  {}

  // get number of threads
  int nthreads ()
  {
    return P;
  }

  // wait at barrier
  void wait (int i)
  {
    // sequential case
    if (P==1) return;
    
    std::unique_lock<std::mutex> ul{mx};
    count += 1; // one more
    if (count<P)
      {
        // wait on my cv until all have arrived
        flag[i] = 1; // indicate I am waiting
        cv[i].wait(ul,[i,this]{return this->flag[i]==0;}); // wait
      }
    else
      {
        // I am the last one, lets wake them up
        count = 0; // reset counter for next turn
        for (int j=0; j<P; j++)
          if (flag[j]==1)
            {
              flag[j] = 0; // the event
              cv[j].notify_one(); // wake up
            }
      }
  }
};  

#endif
