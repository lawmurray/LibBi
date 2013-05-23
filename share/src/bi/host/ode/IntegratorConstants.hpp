/**
 * @file
 *
 * Global variables for ODE integrators.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_ODE_INTEGRATORCONSTANTS_HPP
#define BI_HOST_ODE_INTEGRATORCONSTANTS_HPP

#include "../../math/scalar.hpp"

/**
 * @internal
 *
 * Initial step size.
 */
extern real h_h0;

/**
 * @internal
 *
 * Relative error tolerance.
 */
extern real h_rtoler;

/**
 * @internal
 *
 * Absolute error tolerance.
 */
extern real h_atoler;

/**
 * @internal
 *
 * Rounding unit. Smallest number satisfying 1.0 + uround > 1.0.
 */
extern real h_uround;

/**
 * @internal
 *
 * Safety factor in step size prediction.
 */
extern real h_safe;

/**
 * @internal
 *
 * facl, facr--parameters for step size selection; the new step size is chosen
 * subject to the restriction  facl <= hnew/hold <= facr.
 * Default values are facl = 0.2 and facr = 10.0.
 */
extern real h_facl;

/**
 * @internal
 *
 * @copydoc facl
 */
extern real h_facr;

/**
 * @internal
 *
 * Maximum number of steps before prematurely ending integration.
 */
extern int h_nsteps;

/**
 * @internal
 *
 * The "beta" for stabilized step size control (see section IV.2 of
 * Hairer and Wanner's book). Larger values for beta ( <= 0.1 ) make
 * the step size control more stable. This program needs a larger
 * beta than Higham & Hall. Negative initial value provoke beta = 0.0.
 */
extern real h_beta;

/*
 * Precalculations.
 */
extern real h_expo1;
extern real h_expo;
extern real h_facc1;
extern real h_facc2;
extern real h_logsafe;
extern real h_safe1;

/**
 * Initialise ODE configuration to default.
 *
 * @ingroup method_updater
 */
void h_ode_init();

/**
 * Set ODE initial step size.
 *
 * @ingroup method_updater
 */
void h_ode_set_h0(const real h0in);

/**
 * Set ODE relative error tolerance.
 *
 * @ingroup method_updater
 */
void h_ode_set_rtoler(const real rtolerin);

/**
 * Set ODE absolute error tolerance.
 *
 * @ingroup method_updater
 */
void h_ode_set_atoler(const real atolerin);

/**
 * Set ODE rounding unit. Smallest number satisfying 1.0 + uround > 1.0.
 *
 * @ingroup method_updater
 */
void h_ode_set_uround(const real uroundin);

/**
 * Set ODE safety factor in step size prediction.
 *
 * @ingroup method_updater
 */
void h_ode_set_safe(const real safein);

/**
 * Set the "beta" for stabilized step size control (see section IV.2 of
 * Hairer and Wanner's book). Larger values for beta (<= 0.1) make
 * the step size control more stable. This program needs a larger
 * beta than Higham & Hall. Negative initial value provoke beta = 0.0.
 *
 * @ingroup method_updater
 */
void h_ode_set_beta(const real betain);

/**
 * facl, facr--parameters for step size selection; the new step size is chosen
 * subject to the restriction  facl <= hnew/hold <= facr.
 * Default values are facl = 0.2 and facr = 10.0.
 *
 * @ingroup method_updater
 */
void h_ode_set_facl(const real faclin);

/**
 * @copydoc ode_set_facl
 *
 * @ingroup method_updater
 */
void h_ode_set_facr(const real facrin);

/**
 * Set maximum number of steps before prematurely ending integration.
 *
 * @ingroup method_updater
 */
void h_ode_set_nsteps(const int nstepsin);

#ifdef __CUDACC__
#include "../../cuda/ode/IntegratorConstants.cuh"
#endif

#endif
