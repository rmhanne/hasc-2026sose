#include <iostream>
#include <thread>
#include <atomic>
#include <chrono>
#include <utility>

int n=50000000; // how many times ech process passes
int in1=0;   // indicates that thread 1 is in critical section
int in2=0;   // indicates that thread 2 is in critical section
int last=1;  // process that arrived last at critical section
int count=0; // counts how many times the critical section was passed
std::atomic<int> alast{1};
std::atomic_flag f{ATOMIC_FLAG_INIT};
std::atomic<int> acount{0};

void P1_nolock ()
{
  for (int i=0; i<n; i++)
    count += 1;
}

void P2_nolock ()
{
  for (int i=0; i<n; i++)
    count += 1;
}

void P1_peterson ()
{
  for (int i=0; i<n; i++)
    {
      in1 = 1;     // I want to enter
      last = 1;    // tie breaker
      while (in2 && last==1) ; // busy wait
      count += 1;  // increment counter in c.s.
      in1 = 0;     // we are done
    }
}

void P2_peterson ()
{
  for (int i=0; i<n; i++)
    {
      in2 = 1;     // I want to enter
      last = 2;    // tie breaker
      while (in1 && last==2) ; // busy wait
      count += 1;  // increment counter in c.s.
      in2 = 0;     // we are done
    }
}

void P1_peterson_fenced ()
{
  for (int i=0; i<n; i++)
    {
      in1 = 1;     // I want to enter
      alast.store(1);    // tie breaker
      std::atomic_thread_fence(std::memory_order_release);
      while (in2 && alast.load()==1) ;
      count += 1;  // increment counter in c.s.
      in1 = 0;     // we are done
    }
}

void P2_peterson_fenced ()
{
  for (int i=0; i<n; i++)
    {
      in2 = 1;     // I want to enter
      alast.store(2);    // tie breaker
      std::atomic_thread_fence(std::memory_order_release);
      while (in1 && alast.load()==2) ;
      count += 1;  // increment counter in c.s.
      in2 = 0;     // we are done
    }
}

void P1_atomic_flag ()
{
  for (int i=0; i<n; i++)
    {
      while (f.test_and_set()) ;
      count += 1;  // increment counter in c.s.
      f.clear();   // we are done
    }
}

void P2_atomic_flag ()
{
  for (int i=0; i<n; i++)
    {
      while (f.test_and_set()) ;
      count += 1;  // increment counter in c.s.
      f.clear();   // we are done
    }
}

void P1_atomic ()
{
  for (int i=0; i<n; i++)
    acount++;
}

void P2_atomic ()
{
  for (int i=0; i<n; i++)
    acount++;
}

int main ()
{
  auto start = std::chrono::high_resolution_clock::now();
  std::thread t1a{P1_nolock};
  std::thread t2a{P2_nolock};
  t1a.join();
  t2a.join();
  auto stop = std::chrono::high_resolution_clock::now();
  auto d = stop-start;
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(d).count();
  std::cout << "no_lock        : count is " << count << " and should be " << 2*n << " duration " << (1.0*duration)/n << " us" << std::endl;
 
  count=0;
  start = std::chrono::high_resolution_clock::now();
  std::thread t1b{P1_peterson};
  std::thread t2b{P2_peterson};
  t1b.join();
  t2b.join();
  stop = std::chrono::high_resolution_clock::now();
  d = stop-start;
  duration = std::chrono::duration_cast<std::chrono::microseconds>(d).count();
  std::cout << "peterson       : count is " << count << " and should be " << 2*n << " duration " << (1.0*duration)/n << " us" << std::endl;

  count=0;
  start = std::chrono::high_resolution_clock::now();
  std::thread t1c{P1_peterson_fenced};
  std::thread t2c{P2_peterson_fenced};
  t1c.join();
  t2c.join();
  stop = std::chrono::high_resolution_clock::now();
  d = stop-start;
  duration = std::chrono::duration_cast<std::chrono::microseconds>(d).count();
  std::cout << "peterson_fenced: count is " << count << " and should be " << 2*n << " duration " << (1.0*duration)/n << " us" << std::endl;

  count=0;
  start = std::chrono::high_resolution_clock::now();
  std::thread t1d{P1_atomic_flag};
  std::thread t2d{P2_atomic_flag};
  t1d.join();
  t2d.join();
  stop = std::chrono::high_resolution_clock::now();
  d = stop-start;
  duration = std::chrono::duration_cast<std::chrono::microseconds>(d).count();
  std::cout << "peterson_atomic_flag: count is " << count << " and should be " << 2*n << " duration " << (1.0*duration)/n << " us" << std::endl;

  acount.store(0);
  start = std::chrono::high_resolution_clock::now();
  std::thread t1e{P1_atomic};
  std::thread t2e{P2_atomic};
  t1e.join();
  t2e.join();
  stop = std::chrono::high_resolution_clock::now();
  d = stop-start;
  duration = std::chrono::duration_cast<std::chrono::microseconds>(d).count();
  std::cout << "peterson_atomic: count is " << acount.load() << " and should be " << 2*n << " duration " << (1.0*duration)/n << " us" << std::endl;
}
