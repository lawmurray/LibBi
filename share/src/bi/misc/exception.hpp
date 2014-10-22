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
   * Eigendecomposition failed.
   */
  struct EigenException {
    /**
     * Constructor.
     */
    EigenException(const int info) : info(info) {
      //
    }

    /**
     * Info return by potrf().
     */
    int info;
  };

  /**
   * Particle filter degenerated.
   */
  struct ParticleFilterDegeneratedException {
    //
  };
}

#endif
