/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_METHOD_FILTERFACTORY_HPP
#define BI_METHOD_FILTERFACTORY_HPP

#include "Filter.hpp"
#include "BootstrapPF.hpp"
#include "LookaheadPF.hpp"
#include "BridgePF.hpp"
#include "AdaptivePF.hpp"
#include "ExtendedKF.hpp"

namespace bi {
/**
 * Filter factory.
 *
 * @ingroup method_filter
 */
class FilterFactory {
public:
  /**
   * Create bootstrap particle filter.
   */
  template<class B, class F, class O, class R>
  static Filter<BootstrapPF<B,F,O,R> >* createBootstrapPF(B& m, F& in, O& obs,
      R& resam);

  /**
   * Create lookahead particle filter.
   */
  template<class B, class F, class O, class R>
  static Filter<LookaheadPF<B,F,O,R> >* createLookaheadPF(B& m, F& in, O& obs,
      R& resam);

  /**
   * Create bridge particle filter.
   */
  template<class B, class F, class O, class R>
  static Filter<BridgePF<B,F,O,R> >* createBridgePF(B& m, F& in, O& obs,
      R& resam);

  /**
   * Create adaptive particle filter.
   */
  template<class B, class F, class O, class R, class S2>
  static Filter<AdaptivePF<B,F,O,R,S2> >* createAdaptivePF(B& m, F& in,
      O& obs, R& resam, S2& stopper, const int initialP, const int blockP);

  /**
   * Create extended Kalman filter.
   */
  template<class B, class F, class O>
  static Filter<ExtendedKF<B,F,O> >* createExtendedKF(B& m, F& in, O& obs);
};
}

template<class B, class F, class O, class R>
bi::Filter<bi::BootstrapPF<B,F,O,R> >* bi::FilterFactory::createBootstrapPF(
    B& m, F& in, O& obs, R& resam) {
  return new Filter<BootstrapPF<B,F,O,R> >(m, in, obs, resam);
}

template<class B, class F, class O, class R>
bi::Filter<bi::LookaheadPF<B,F,O,R> >* bi::FilterFactory::createLookaheadPF(
    B& m, F& in, O& obs, R& resam) {
  return new Filter<LookaheadPF<B,F,O,R> >(m, in, obs, resam);
}

template<class B, class F, class O, class R>
bi::Filter<bi::BridgePF<B,F,O,R> >* bi::FilterFactory::createBridgePF(B& m,
    F& in, O& obs, R& resam) {
  return new Filter<BridgePF<B,F,O,R> >(m, in, obs, resam);
}

template<class B, class F, class O, class R, class S2>
bi::Filter<bi::AdaptivePF<B,F,O,R,S2> >* bi::FilterFactory::createAdaptivePF(
    B& m, F& in, O& obs, R& resam, S2& stopper, const int initialP,
    const int blockP) {
  return new Filter<AdaptivePF<B,F,O,R,S2> >(m, in, obs, resam, stopper,
      initialP, blockP);
}

template<class B, class F, class O>
bi::Filter<bi::ExtendedKF<B,F,O> >* bi::FilterFactory::createExtendedKF(B& m,
    F& in, O& obs) {
  return new Filter<ExtendedKF<B,F,O> >(m, in, obs);
}

#endif
