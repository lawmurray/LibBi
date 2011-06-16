/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "IntegratorConstants.hpp"

#include "../math/scalar.hpp"
#include "../cuda/cuda.hpp"

real h_h0;
real h_rtoler;
real h_atoler;
real h_uround;
real h_safe;
real h_facl;
real h_facr;
int h_nsteps;
real h_beta;
real h_expo1;
real h_expo;
real h_facc1;
real h_facc2;
real h_logsafe;
real h_safe1;

void h_ode_set_h0(const real h0in) {
  h_h0 = h0in;
}

void h_ode_set_rtoler(const real rtolerin) {
  h_rtoler = rtolerin;
}

void h_ode_set_atoler(const real atolerin) {
  h_atoler = atolerin;
}

void h_ode_set_uround(const real uroundin) {
  /* pre-condition */
  assert (uroundin > REAL(1.0e-19) && uroundin < REAL(1.0));

  h_uround = uroundin;
}

void h_ode_set_safe(const real safein) {
  /* pre-condition */
  assert (safein > REAL(0.001) && safein < REAL(1.0));

  h_safe = safein;
  h_safe1 = REAL(1.0) / safein;
  h_logsafe = CUDA_LOG(safein);
}

void h_ode_set_beta(const real betain) {
  /* pre-condition */
  assert (betain >= 0.0 && betain <= REAL(0.2));

  h_beta = betain;
  h_expo1 = REAL(0.2) - betain*REAL(0.75);
  h_expo = REAL(0.5)*(REAL(0.2) - betain*REAL(0.75));
}

void h_ode_set_facl(const real faclin) {
  h_facl = faclin;
  h_facc1 = REAL(1.0) / faclin;
}

void h_ode_set_facr(const real facrin) {
  h_facr = facrin;
  h_facc2 = REAL(1.0) / facrin;
}

void h_ode_set_nsteps(const int nstepsin) {
  h_nsteps = nstepsin;
}

void h_ode_init() {
  h_ode_set_h0(REAL(1.0e-2));
  h_ode_set_rtoler(REAL(1.0e-7));
  h_ode_set_atoler(REAL(1.0e-7));
  h_ode_set_uround(REAL(1.0e-16));
  h_ode_set_safe(REAL(0.9));
  h_ode_set_facl(REAL(0.2));
  h_ode_set_facr(REAL(10.0));
  h_ode_set_beta(REAL(0.04));
  h_ode_set_nsteps(1000u);
}
