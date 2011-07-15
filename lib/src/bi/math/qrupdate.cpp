/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev: 1244 $
 * $Date: 2011-01-31 10:37:29 +0800 (Mon, 31 Jan 2011) $
 */
#include "qrupdate.hpp"

#define QRUPDATE_FUNC_DEF(name, dname, sname) \
  BOOST_TYPEOF(sname##_) *bi::qrupdate_##name<float>::func = sname##_; \
  BOOST_TYPEOF(dname##_) *bi::qrupdate_##name<double>::func = dname##_;

QRUPDATE_FUNC_DEF(ch1up, dch1up, sch1up)
QRUPDATE_FUNC_DEF(ch1dn, dch1dn, sch1dn)
