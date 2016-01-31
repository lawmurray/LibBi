/**
 * @file
 *
 * @author Lawrence Murray <murray@stats.ox.ac.uk>
 */
#ifndef BI_ADAPTER_ADAPTER_HPP
#define BI_ADAPTER_ADAPTER_HPP

namespace bi {
/**
 * Adapter.
 *
 * @ingroup method_adapter
 */
template<class A>
class Adapter: public A {
public:
  /**
   * Constructor.
   */
  Adapter(const bool local = false, const double scale = 0.25);
};
}

template<class A>
bi::Adapter<A>::Adapter(const bool local, const double scale) :
    A(local, scale) {
  //
}

#endif
