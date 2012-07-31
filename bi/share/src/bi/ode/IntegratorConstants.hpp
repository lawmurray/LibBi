/**
 * @file
 *
 * Easy interface for setting ODE parameters.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 */
#ifndef BI_MATH_ODE_HPP
#define BI_MATH_ODE_HPP

#include "../host/ode/IntegratorConstants.hpp"
#ifdef __CUDACC__
#include "../cuda/ode/IntegratorConstants.cuh"
#endif

/**
 * Initialise ODE parameters.
 */
void bi_ode_init();

/**
 * Set ODE parameters.
 *
 * @param h0 Initial step size.
 * @param atoler Absolute error tolerance.
 * @param rtoler Relative error tolerance.
 */
void bi_ode_set(const real h0, const real atoler, const real rtoler);

inline void bi_ode_init() {
  #ifdef __CUDACC__
  ode_init();
  #endif
  h_ode_init();
}

inline void bi_ode_set(const real h0, const real atoler, const real rtoler) {
  if (h_h0 != h0) {
    h_ode_set_h0(h0);
    #ifdef __CUDACC__
    ode_set_h0(h0);
    #endif
  }
  if (h_atoler != atoler) {
    h_ode_set_atoler(atoler);
    #ifdef __CUDACC__
    ode_set_atoler(atoler);
    #endif
  }
  if (h_rtoler != rtoler) {
    h_ode_set_rtoler(rtoler);
    #ifdef __CUDACC__
    ode_set_rtoler(rtoler);
    #endif
  }
}

#endif
