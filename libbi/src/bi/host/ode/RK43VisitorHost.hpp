/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_HOST_ODE_RK43VISITORHOST_HPP
#define BI_HOST_ODE_RK43VISITORHOST_HPP

#include "../../ode/RK43Stage.hpp"

namespace bi {
/**
 * Visitor for RK43Integrator.
 *
 * @tparam B Model type.
 * @tparam S1 Action type list.
 * @tparam S2 Action type list.
 * @tparam T1 Scalar type.
 * @tparam PX Parents type.
 * @tparam T2 Scalar type.
 */
template<class B, class S1, class S2, class T1, class PX, class T2>
class RK43VisitorHost {
public:
  static void stage1(const T1 t, const T1 h, const int p, const PX& pax,
      T2* r1, T2* r2, T2* err) {
    coord_type cox;
    int id = start;

    while (id < end) {
      stage::stage1(t, h, p, cox, pax, r1[id], r2[id], err[id]);
      ++cox;
      ++id;
    }
    visitor::stage1(t, h, p, pax, r1, r2, err);
  }

  static void stage2(const T1 t, const T1 h, const int p, const PX& pax,
      T2* r1, T2* r2, T2* err) {
    coord_type cox;
    int id = start;

    while (id < end) {
      stage::stage2(t, h, p, cox, pax, r1[id], r2[id], err[id]);
      ++cox;
      ++id;
    }
    visitor::stage2(t, h, p, pax, r1, r2, err);
  }

  static void stage3(const T1 t, const T1 h, const int p, const PX& pax,
      T2* r1, T2* r2, T2* err) {
    coord_type cox;
    int id = start;

    while (id < end) {
      stage::stage3(t, h, p, cox, pax, r1[id], r2[id], err[id]);
      ++cox;
      ++id;
    }
    visitor::stage3(t, h, p, pax, r1, r2, err);
  }

  static void stage4(const T1 t, const T1 h, const int p, const PX& pax,
      T2* r1, T2* r2, T2* err) {
    coord_type cox;
    int id = start;

    while (id < end) {
      stage::stage4(t, h, p, cox, pax, r1[id], r2[id], err[id]);
      ++cox;
      ++id;
    }
    visitor::stage4(t, h, p, pax, r1, r2, err);
  }

  static void stage5(const T1 t, const T1 h, const int p, const PX& pax,
      T2* r1, T2* r2, T2* err) {
    coord_type cox;
    int id = start;

    while (id < end) {
      stage::stage5(t, h, p, cox, pax, r1[id], r2[id], err[id]);
      ++cox;
      ++id;
    }
    visitor::stage5(t, h, p, pax, r1, r2, err);
  }

private:
  typedef typename front<S2>::type front;
  typedef typename pop_front<S2>::type pop_front;
  typedef typename front::target_type target_type;
  typedef typename target_type::coord_type coord_type;

  typedef RK43Stage<front,T1,coord_type,PX,T2> stage;
  typedef RK43VisitorHost<B,S1,pop_front,T1,PX,T2> visitor;

  static const int start = target_start<S1,target_type>::value;
  static const int end = target_end<S1,target_type>::value;
};

/**
 * @internal
 *
 * Base case of RK43Visitor.
 */
template<class B, class S1, class T1, class PX, class T2>
class RK43VisitorHost<B,S1,empty_typelist,T1,PX,T2> {
public:
  static void stage1(const T1 t, const T1 h, const int p, const PX& pax,
      T2* r1, T2* r2, T2* err) {
    //
  }

  static void stage2(const T1 t, const T1 h, const int p, const PX& pax,
      T2* r1, T2* r2, T2* err) {
    //
  }

  static void stage3(const T1 t, const T1 h, const int p, const PX& pax,
      T2* r1, T2* r2, T2* err) {
    //
  }

  static void stage4(const T1 t, const T1 h, const int p, const PX& pax,
      T2* r1, T2* r2, T2* err) {
    //
  }

  static void stage5(const T1 t, const T1 h, const int p, const PX& pax,
      T2* r1, T2* r2, T2* err) {
    //
  }
};

}

#endif
