/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_MISC_EXCEPTION_HPP
#define BI_MISC_EXCEPTION_HPP

namespace bi {
  /**
   * Exceptions.
   */
  enum Exception {
    /**
     * Cholesky decomposition failed (matrix not positive definite).
     */
    CHOLESKY_FAILED = 101
  };
}

#endif
