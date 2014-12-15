/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_ADAPTER_MARGINALSIRADAPTER_HPP
#define BI_ADAPTER_MARGINALSIRADAPTER_HPP

#include "GaussianAdapter.hpp"

namespace bi {
/**
 * Adapter for MarginalSIR.
 *
 * @ingroup method_adapter
 */
template<class B, Location L>
class MarginalSIRAdapter: public GaussianAdapter<B,L> {
public:
  /**
   * Constructor.
   */
  MarginalSIRAdapter(const double essRel = 0.5, const bool local = false, const double scale = 0.5);

  /**
   * Add state.
   */
  template<class S1>
  void add(const S1& s);
};
}

template<class B, bi::Location L>
bi::MarginalSIRAdapter<B,L>::MarginalSIRAdapter(const double essRel, const bool local, const double scale) :
    GaussianAdapter<B,L>(essRel, local, scale) {
  //
}

template<class B, bi::Location L>
template<class S1>
void bi::MarginalSIRAdapter<B,L>::add(const S1& s) {
  for (int p = 0; p < s.size(); ++p) {
    GaussianAdapter<B,L>::add(*s.s1s[p], subrange(s.logWeights(), p, 1));
  }
}

#endif
