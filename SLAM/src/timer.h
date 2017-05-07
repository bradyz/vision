#ifndef TIMER_H
#define TIMER_H

#include <cmath>
#include <ctime>

// All times are in milliseconds.
class Timer {

public:
  Timer() : start_(std::clock()), tick_(std::clock()) { }

  double lap() {
    double result = diff(tick_, std::clock());
    tick_ = std::clock();
    return result;
  }

  double elapsed() const {
    return diff(start_, std::clock());
  }

  static double diff(std::clock_t t1, std::clock_t t2) {
    return std::fabs(t2 - t1) / CLOCKS_PER_SEC * 1000.0;
  }

private:
  std::clock_t start_;
  std::clock_t tick_;

};

#endif
