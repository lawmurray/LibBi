/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ADAPTER_MARGINALMHADAPTER_HPP
#define BI_ADAPTER_MARGINALMHADAPTER_HPP

#include "GaussianAdapter.hpp"

namespace bi {
/**
 * Adapter for MarginalMH.
 *
 * @ingroup method_adapter
 */
template<class B, Location L>
class MarginalMHAdapter: public GaussianAdapter<B,L> {
public:
  /**
   * Constructor.
   */
  MarginalMHAdapter(const double essRel = 0.5, const bool local = false, const double scale = 0.5);

  /**
   * Add state.
   */
  template<class S1>
  void add(const S1& s);
};
}

template<class B, bi::Location L>
bi::MarginalMHAdapter<B,L>::MarginalMHAdapter(const double essRel, const bool local, const double scale) :
    GaussianAdapter<B,L>(essRel, local, scale) {
  //
}

template<class B, bi::Location L>
template<class S1>
void bi::MarginalMHAdapter<B,L>::add(const S1& s) {

}

#endif
