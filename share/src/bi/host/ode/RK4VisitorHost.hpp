/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_ODE_RK4VISITORHOST_HPP
#define BI_HOST_ODE_RK4VISITORHOST_HPP

#include "../../ode/RK4Stage.hpp"

namespace bi {
/**
 * Visitor for RK4Integrator.
 *
 * @tparam B Model type.
 * @tparam S1 Action type list.
 * @tparam S2 Action type list.
 * @tparam T1 Scalar type.
 * @tparam PX Parents type.
 * @tparam T2 Scalar type.
 */
template<class B, class S1, class S2, class T1, class PX, class T2>
class RK4VisitorHost {
public:
  static void stage1(const T1 t, const T1 h, const State<B,ON_HOST>& s,
      const int p, const PX& pax, const T2* x0, T2* x1, T2* x2, T2* x3,
      T2* x4) {
    coord_type cox;
    int id = start;

    while (id < end) {
      stage::stage1(t, h, s, p, cox, pax, x0[id], x1[id], x2[id], x3[id],
          x4[id]);
      ++cox;
      ++id;
    }
    visitor::stage1(t, h, s, p, pax, x0, x1, x2, x3, x4);
  }

  static void stage2(const T1 t, const T1 h, const State<B,ON_HOST>& s,
      const int p, const PX& pax, const T2* x0, T2* x2, T2* x3, T2* x4) {
    coord_type cox;
    int id = start;

    while (id < end) {
      stage::stage2(t, h, s, p, cox, pax, x0[id], x2[id], x3[id], x4[id]);
      ++cox;
      ++id;
    }
    visitor::stage2(t, h, s, p, pax, x0, x2, x3, x4);
  }

  static void stage3(const T1 t, const T1 h, const State<B,ON_HOST>& s,
      const int p, const PX& pax, const T2* x0, T2* x3, T2* x4) {
    coord_type cox;
    int id = start;

    while (id < end) {
      stage::stage3(t, h, s, p, cox, pax, x0[id], x3[id], x4[id]);
      ++cox;
      ++id;
    }
    visitor::stage3(t, h, s, p, pax, x0, x3, x4);
  }

  static void stage4(const T1 t, const T1 h, const State<B,ON_HOST>& s,
      const int p, const PX& pax, const T2* x0, T2* x4) {
    coord_type cox;
    int id = start;

    while (id < end) {
      stage::stage4(t, h, s, p, cox, pax, x0[id], x4[id]);
      ++cox;
      ++id;
    }
    visitor::stage4(t, h, s, p, pax, x0, x4);
  }

private:
  typedef typename front<S2>::type front;
  typedef typename pop_front<S2>::type pop_front;
  typedef typename front::coord_type coord_type;

  typedef RK4Stage<front,T1,B,ON_HOST,coord_type,PX,T2> stage;
  typedef RK4VisitorHost<B,S1,pop_front,T1,PX,T2> visitor;

  static const int start = action_start<S1,front>::value;
  static const int end = action_end<S1,front>::value;
};

/**
 * @internal
 *
 * Base case of RK4Visitor.
 */
template<class B, class S1, class T1, class PX, class T2>
class RK4VisitorHost<B,S1,empty_typelist,T1,PX,T2> {
public:
  static void stage1(const T1 t, const T1 h, const State<B,ON_HOST>& s,
      const int p, const PX& pax, const T2* x0, T2* x1, T2* x2, T2* x3,
      T2* x4) {
    //
  }

  static void stage2(const T1 t, const T1 h, const State<B,ON_HOST>& s,
      const int p, const PX& pax, const T2* x0, T2* x2, T2* x3, T2* x4) {
    //
  }

  static void stage3(const T1 t, const T1 h, const State<B,ON_HOST>& s,
      const int p, const PX& pax, const T2* x0, T2* x3, T2* x4) {
    //
  }

  static void stage4(const T1 t, const T1 h, const State<B,ON_HOST>& s,
      const int p, const PX& pax, const T2* x0, T2* x4) {
    //
  }
};

}

#endif
