/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_SSE_SSESTATE_HPP
#define BI_SSE_SSESTATE_HPP

#include "sse.hpp"

namespace bi {
/**
 * @internal
 *
 * Get variable on host.
 *
 * @tparam M1 Matrix type.
 *
 * @param s State.
 * @param p First trajectory id.
 * @param id Node id.
 *
 * @return Values of node @p id for the @c BI_SSE_SIZE trajectories starting
 * at @p p.
 */
template<class M1>
sse_real sse_state_get(const M1& s, const int p, const int id);

/**
 * @internal
 *
 * Set variable on host.
 *
 * @tparam M1 Matrix type.
 *
 * @param s State.
 * @param p Trajectory id.
 * @param id Node id.
 * @param[out] val Value to set.
 *
 * Sets the given variable to the given value.
 */
template<class M1>
void sse_state_set(M1& s, const int p, const int id, const sse_real& val);

}

template<class M1>
BI_FORCE_INLINE inline bi::sse_real bi::sse_state_get(const M1& s,
    const int p, const int id) {
  /* pre-condition */
  assert (p % BI_SSE_SIZE == 0);

  if (s.size1() == 1) {
    return sse_real(s(p, id));
  } else {
    return sse_real(&s(p, id));
  }
}

template<class M1>
BI_FORCE_INLINE inline void bi::sse_state_set(M1& s,
    const int p, const int id, const sse_real& val) {
  /* pre-condition */
  assert (p % BI_SSE_SIZE == 0);

  if (s.size1() == 1) {
    val.store(s(p, id));
  } else {
    val.store(&s(p, id));
  }
}

#endif
