/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#include "qrupdate.hpp"

#define QRUPDATE_FUNC_DEF(name, dname, sname) \
  BOOST_TYPEOF(sname##_) *bi::qrupdate_##name<float>::func = sname##_; \
  BOOST_TYPEOF(dname##_) *bi::qrupdate_##name<double>::func = dname##_;

QRUPDATE_FUNC_DEF(ch1up, dch1up, sch1up)
QRUPDATE_FUNC_DEF(ch1dn, dch1dn, sch1dn)
