/**
 * @file
 *
 * Global variables for ODE integrators.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ODE_INTEGRATORCONSTANTS_HPP
#define BI_ODE_INTEGRATORCONSTANTS_HPP

#include "../cuda/cuda.hpp"
#include "../math/scalar.hpp"

//#define h0 REAL(1.0)
//#define rtoler REAL(1.0e-3)
//#define atoler REAL(1.0e-3)
//#define uround REAL(1.0e-16)
//#define safe REAL(0.9)
//#define facl REAL(0.2)
//#define facr REAL(10.0)
//#define beta REAL(0.04)
//#define nsteps 1000
//#define safe1 (REAL(1.0) / safe)
//#define logsafe CUDA_LOG(safe)
//#define expo1 (REAL(0.2) - beta*REAL(0.75));
//#define expo (REAL(0.5)*(REAL(0.2) - beta*REAL(0.75)))
//#define facc1 (REAL(1.0) / facl)
//#define facc2 (REAL(1.0) / facr)
//
//#define h_h0 REAL(1.0)
//#define h_rtoler REAL(1.0e-3)
//#define h_atoler REAL(1.0e-3)
//#define h_uround REAL(1.0e-16)
//#define h_safe REAL(0.9)
//#define h_facl REAL(0.2)
//#define h_facr REAL(10.0)
//#define h_beta REAL(0.04)
//#define h_nsteps 1000
//#define h_safe1 (REAL(1.0) / safe)
//#define h_logsafe CUDA_LOG(safe)
//#define h_expo1 (REAL(0.2) - beta*REAL(0.75));
//#define h_expo (REAL(0.5)*(REAL(0.2) - beta*REAL(0.75)))
//#define h_facc1 (REAL(1.0) / facl)
//#define h_facc2 (REAL(1.0) / facr)

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
void h_ode_set_h0(real h0in);

/**
 * Set ODE relative error tolerance.
 *
 * @ingroup method_updater
 */
void h_ode_set_rtoler(real rtolerin);

/**
 * Set ODE absolute error tolerance.
 *
 * @ingroup method_updater
 */
void h_ode_set_atoler(real atolerin);

/**
 * Set ODE rounding unit. Smallest number satisfying 1.0 + uround > 1.0.
 *
 * @ingroup method_updater
 */
void h_ode_set_uround(real uroundin);

/**
 * Set ODE safety factor in step size prediction.
 *
 * @ingroup method_updater
 */
void h_ode_set_safe(real safein);

/**
 * Set the "beta" for stabilized step size control (see section IV.2 of
 * Hairer and Wanner's book). Larger values for beta (<= 0.1) make
 * the step size control more stable. This program needs a larger
 * beta than Higham & Hall. Negative initial value provoke beta = 0.0.
 *
 * @ingroup method_updater
 */
void h_ode_set_beta(real betain);

/**
 * facl, facr--parameters for step size selection; the new step size is chosen
 * subject to the restriction  facl <= hnew/hold <= facr.
 * Default values are facl = 0.2 and facr = 10.0.
 *
 * @ingroup method_updater
 */
void h_ode_set_facl(real faclin);

/**
 * @copydoc ode_set_facl
 *
 * @ingroup method_updater
 */
void h_ode_set_facr(real facrin);

/**
 * Set maximum number of steps before prematurely ending integration.
 *
 * @ingroup method_updater
 */
void h_ode_set_nsteps(int nsteps);

#ifdef __CUDACC__
#include "../cuda/ode/IntegratorConstants.cuh"
#endif

#endif
