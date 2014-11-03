/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_LOOP_HPP
#define BI_PROGRAM_LOOP_HPP

#include "Conditioned.hpp"
#include "Braced.hpp"
#include "Expression.hpp"

namespace biprog {
/**
 * Loop.
 *
 * @ingroup program
 */
class Loop: public Conditioned,
    public Braced,
    public boost::enable_shared_from_this<Loop> {
public:
  /**
   * Constructor.
   */
  Loop(boost::shared_ptr<Expression> cond,
      boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~Loop();

  /*
   * Operators.
   */
  using Expression::operator<;
  using Expression::operator<=;
  using Expression::operator>;
  using Expression::operator>=;
  using Expression::operator==;
  using Expression::operator!=;
  virtual bool operator<(const Loop& o) const;
  virtual bool operator<=(const Loop& o) const;
  virtual bool operator>(const Loop& o) const;
  virtual bool operator>=(const Loop& o) const;
  virtual bool operator==(const Loop& o) const;
  virtual bool operator!=(const Loop& o) const;
};
}

inline biprog::Loop::Loop(boost::shared_ptr<Expression> cond,
    boost::shared_ptr<Expression> braces) :
    Conditioned(cond), Braced(braces) {
  //
}

inline biprog::Loop::~Loop() {
  //
}

inline bool biprog::Loop::operator<(const Loop& o) const {
  return *cond < *o.cond && *braces < *o.braces;
}

inline bool biprog::Loop::operator<=(const Loop& o) const {
  return *cond <= *o.cond && *braces <= *o.braces;
}

inline bool biprog::Loop::operator>(const Loop& o) const {
  return *cond > *o.cond && *braces > *o.braces;
}

inline bool biprog::Loop::operator>=(const Loop& o) const {
  return *cond >= *o.cond && *braces >= *o.braces;
}

inline bool biprog::Loop::operator==(const Loop& o) const {
  return *cond == *o.cond && *braces == *o.braces;
}

inline bool biprog::Loop::operator!=(const Loop& o) const {
  return *cond != *o.cond || *braces != *o.braces;
}

#endif
