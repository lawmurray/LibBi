/**
 * @file
 *
 * Global variables, held in constant device memory, for ODE integrators.
 * Adapted from IntegratorT from Blake Ashby <bmashby@stanford.edu>, see
 * NonStiffIntegrator there.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_ODE_INTEGRATORCONSTANTS_CUH
#define BI_CUDA_ODE_INTEGRATORCONSTANTS_CUH

#include "../cuda.hpp"
#include "../../math/scalar.hpp"

/**
 * @internal
 *
 * Initial step size.
 */
static CUDA_VAR_CONSTANT real h0;

/**
 * @internal
 *
 * Relative error tolerance.
 */
static CUDA_VAR_CONSTANT real rtoler;

/**
 * @internal
 *
 * Absolute error tolerance.
 */
static CUDA_VAR_CONSTANT real atoler;

/**
 * @internal
 *
 * Rounding unit. Smallest number satisfying 1.0 + uround > 1.0.
 */
static CUDA_VAR_CONSTANT real uround;

/**
 * @internal
 *
 * Safety factor in step size prediction.
 */
static CUDA_VAR_CONSTANT real safe;

/**
 * @internal
 *
 * facl, facr--parameters for step size selection; the new step size is chosen
 * subject to the restriction  facl <= hnew/hold <= facr.
 * Default values are facl = 0.2 and facr = 10.0.
 */
static CUDA_VAR_CONSTANT real facl;

/**
 * @internal
 *
 * @copydoc facl
 */
static CUDA_VAR_CONSTANT real facr;

/**
 * @internal
 *
 * Maximum number of steps before prematurely ending integration.
 */
static CUDA_VAR_CONSTANT int nsteps;

/**
 * @internal
 *
 * The "beta" for stabilized step size control (see section IV.2 of
 * Hairer and Wanner's book). Larger values for beta ( <= 0.1 ) make
 * the step size control more stable. This program needs a larger
 * beta than Higham & Hall. Negative initial value provoke beta = 0.0.
 */
static CUDA_VAR_CONSTANT real beta;

/*
 * Precalculations.
 */
static CUDA_VAR_CONSTANT real expo1;
static CUDA_VAR_CONSTANT real expo;
static CUDA_VAR_CONSTANT real facc1;
static CUDA_VAR_CONSTANT real facc2;
static CUDA_VAR_CONSTANT real logsafe;
static CUDA_VAR_CONSTANT real safe1;

/**
 * Initialise ODE configuration to default.
 *
 * @ingroup method_updater
 */
void ode_init();

/**
 * Set ODE initial step size.
 *
 * @ingroup method_updater
 */
void ode_set_h0(real h0in);

/**
 * Set ODE relative error tolerance.
 *
 * @ingroup method_updater
 */
void ode_set_rtoler(real rtolerin);

/**
 * Set ODE absolute error tolerance.
 *
 * @ingroup method_updater
 */
void ode_set_atoler(real atolerin);

/**
 * Set ODE rounding unit. Smallest number satisfying 1.0 + uround > 1.0.
 *
 * @ingroup method_updater
 */
void ode_set_uround(real uroundin);

/**
 * Set ODE safety factor in step size prediction.
 *
 * @ingroup method_updater
 */
void ode_set_safe(real safein);

/**
 * Set the "beta" for stabilized step size control (see section IV.2 of
 * Hairer and Wanner's book). Larger values for beta (<= 0.1) make
 * the step size control more stable. This program needs a larger
 * beta than Higham & Hall. Negative initial value provoke beta = 0.0.
 *
 * @ingroup method_updater
 */
void ode_set_beta(real betain);

/**
 * facl, facr--parameters for step size selection; the new step size is chosen
 * subject to the restriction  facl <= hnew/hold <= facr.
 * Default values are facl = 0.2 and facr = 10.0.
 *
 * @ingroup method_updater
 */
void ode_set_facl(real faclin);

/**
 * @copydoc ode_set_facl
 *
 * @ingroup method_updater
 */
void ode_set_facr(real facrin);

/**
 * Set maximum number of steps before prematurely ending integration.
 *
 * @ingroup method_updater
 */
void ode_set_nsteps(int nstepsin);

inline void ode_set_h0(real h0in) {
  CUDA_SET_CONSTANT(real, h0, h0in);
}

inline void ode_set_rtoler(real rtolerin) {
  CUDA_SET_CONSTANT(real, rtoler, rtolerin);
}

inline void ode_set_atoler(real atolerin) {
  CUDA_SET_CONSTANT(real, atoler, atolerin);
}

inline void ode_set_uround(real uroundin) {
  /* pre-condition */
  BI_ASSERT(uroundin > 1.0e-19 && uroundin < 1.0);

  CUDA_SET_CONSTANT(real, uround, uroundin);
}

inline void ode_set_safe(real safein) {
  /* pre-condition */
  BI_ASSERT(safein > 0.001 && safein < 1.0);
  real val;

  CUDA_SET_CONSTANT(real, safe, safein);
  val = BI_REAL(1.0 / safein);
  CUDA_SET_CONSTANT(real, safe1, val);
  val = BI_REAL(log(safein));
  CUDA_SET_CONSTANT(real, logsafe, val);
}

inline void ode_set_beta(real betain) {
  /* pre-condition */
  BI_ASSERT(betain >= 0.0 && betain <= 0.2);
  real val;

  CUDA_SET_CONSTANT(real, beta, betain);
  val = BI_REAL(0.2 - betain*0.75);
  CUDA_SET_CONSTANT(real, expo1, val);
  val = BI_REAL(0.5*(0.2 - betain*0.75));
  CUDA_SET_CONSTANT(real, expo, val);
}

inline void ode_set_facl(real faclin) {
  real val;

  CUDA_SET_CONSTANT(real, facl, faclin);
  val = BI_REAL(1.0 / faclin);
  CUDA_SET_CONSTANT(real, facc1, val);
}

inline void ode_set_facr(real facrin) {
  real val;

  CUDA_SET_CONSTANT(real, facr, facrin);
  val = BI_REAL(1.0 / facrin);
  CUDA_SET_CONSTANT(real, facc2, val);
}

inline void ode_set_nsteps(int nstepsin) {
  CUDA_SET_CONSTANT(int, nsteps, nstepsin);
}

inline void ode_init() {
  ode_set_h0(BI_REAL(1.0e-2));
  ode_set_rtoler(BI_REAL(1.0e-7));
  ode_set_atoler(BI_REAL(1.0e-7));
  ode_set_uround(BI_REAL(1.0e-16));
  ode_set_safe(BI_REAL(0.9));
  ode_set_facl(BI_REAL(0.2));
  ode_set_facr(BI_REAL(10.0));
  ode_set_beta(BI_REAL(0.04));
  ode_set_nsteps(1000);
}

#endif
