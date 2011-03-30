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

#include "../ode/IntegratorConstants.hpp"

/**
 * Set basic ODE parameters.
 *
 * @param h0 Initial step size.
 * @param atoler Absolute error tolerance.
 * @param rtoler Relative error tolerance.
 */
void bi_ode_init(const real h0, const real atoler, const real rtoler);

inline void bi_ode_init(const real h0, const real atoler, const real rtoler) {
  #ifdef __CUDACC__
  ode_init();
  ode_set_h0(h0);
  ode_set_atoler(atoler);
  ode_set_rtoler(rtoler);
  #endif
  h_ode_init();
  h_ode_set_h0(h0);
  h_ode_set_atoler(atoler);
  h_ode_set_rtoler(rtoler);
}

#endif
