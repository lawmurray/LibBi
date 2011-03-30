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
  int toc();

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

inline int bi::TicToc::toc() {
  timeval end;
  gettimeofday(&end, NULL);

  return (end.tv_sec - start.tv_sec)*1e6 + (end.tv_usec - start.tv_usec);
}

#endif
