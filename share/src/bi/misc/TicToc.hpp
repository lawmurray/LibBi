/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MISC_TICTOC_HPP
#define BI_MISC_TICTOC_HPP

#include <sys/time.h>

namespace bi {
/**
 * Timing class.
 *
 * @ingroup misc
 */
class TicToc {
public:
  /**
   * Constructor. Starts timer.
   */
  TicToc();

  /**
   * Start or restart timer.
   */
  void tic();

  /**
   * Read timer.
   *
   * @return Number of microseconds since last call to tic().
   */
  long toc();

  /**
   * Return absolute time.
   */
  long time();

private:
  /**
   * Time of last call to tic().
   */
  timeval start;
};

}

inline bi::TicToc::TicToc() {
  tic();
}

inline void bi::TicToc::tic() {
  gettimeofday(&start, NULL);
}

inline long bi::TicToc::toc() {
  timeval end;
  gettimeofday(&end, NULL);

  return (end.tv_sec - start.tv_sec)*1e6 + (end.tv_usec - start.tv_usec);
}

inline long bi::TicToc::time() {
  timeval now;
  gettimeofday(&now, NULL);

  return now.tv_sec*1e6 + now.tv_usec;
}

#endif
