/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "IntegratorConstants.hpp"

#include "../../math/function.hpp"

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
  BI_ASSERT(uroundin > BI_REAL(1.0e-19) && uroundin < BI_REAL(1.0));

  h_uround = uroundin;
}

void h_ode_set_safe(const real safein) {
  /* pre-condition */
  BI_ASSERT(safein > BI_REAL(0.001) && safein < BI_REAL(1.0));

  h_safe = safein;
  h_safe1 = BI_REAL(1.0) / safein;
  h_logsafe = bi::log(safein);
}

void h_ode_set_beta(const real betain) {
  /* pre-condition */
  BI_ASSERT(betain >= 0.0 && betain <= BI_REAL(0.2));

  h_beta = betain;
  h_expo1 = BI_REAL(0.2) - betain*BI_REAL(0.75);
  h_expo = BI_REAL(0.5)*(BI_REAL(0.2) - betain*BI_REAL(0.75));
}

void h_ode_set_facl(const real faclin) {
  h_facl = faclin;
  h_facc1 = BI_REAL(1.0) / faclin;
}

void h_ode_set_facr(const real facrin) {
  h_facr = facrin;
  h_facc2 = BI_REAL(1.0) / facrin;
}

void h_ode_set_nsteps(const int nstepsin) {
  h_nsteps = nstepsin;
}

void h_ode_init() {
  h_ode_set_h0(BI_REAL(1.0e-2));
  h_ode_set_rtoler(BI_REAL(1.0e-7));
  h_ode_set_atoler(BI_REAL(1.0e-7));
  h_ode_set_uround(BI_REAL(1.0e-16));
  h_ode_set_safe(BI_REAL(0.9));
  h_ode_set_facl(BI_REAL(0.2));
  h_ode_set_facr(BI_REAL(10.0));
  h_ode_set_beta(BI_REAL(0.04));
  h_ode_set_nsteps(1000);
}
