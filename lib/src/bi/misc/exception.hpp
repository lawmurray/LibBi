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
   * Cholesky decomposition failed (matrix not positive definite).
   */
  struct CholeskyException {
    /**
     * Constructor.
     */
    CholeskyException(const int info) : info(info) {
      //
    }

    /**
     * Info return by potrf().
     */
    int info;
  };

  /**
   * Cholesky downdate failed (matrix not positive definite).
   */
  struct CholeskyDowndateException {
    /**
     * Constructor.
     */
    CholeskyDowndateException(const int info) : info(info) {
      //
    }

    /**
     * Info return by ch1dn().
     */
    int info;
  };
}

#endif
