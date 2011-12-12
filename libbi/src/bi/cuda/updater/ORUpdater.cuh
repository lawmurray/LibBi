/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_CUDA_UPDATER_ORUPDATER_CUH
#define BI_CUDA_UPDATER_ORUPDATER_CUH

#include "../../updater/ORUpdateVisitor.hpp"

template<class B>
void bi::ORUpdater<B>::update(State<bi::ON_DEVICE>& s) {
  typedef typename B::OTypeList S;

  if (net_size<B,S>::value > 0) {
    clean();
    BOOST_AUTO(buf, host_temp_matrix<real>(s.get(OR_NODE).size1(), s.get(OR_NODE).size2()));

    typedef BOOST_TYPEOF(*buf) M1;
    typedef ORUpdateVisitor<B,S,M1> Visitor;

    Visitor::accept(rng, *buf);

    s.get(OR_NODE) = *buf;
    add(buf);
  }
}

#endif
