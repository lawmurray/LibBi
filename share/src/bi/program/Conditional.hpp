/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_CONDITIONAL_HPP
#define BI_PROGRAM_CONDITIONAL_HPP

#include "Conditioned.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Conditional.
 *
 * @ingroup program
 */
class Conditional: public virtual Conditioned,
    public virtual Braced,
    public boost::enable_shared_from_this<Conditional> {
public:
  /**
   * Constructor.
   */
  Conditional(boost::shared_ptr<Expression> cond,
      boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~Conditional();

  /*
   * Operators.
   */
  virtual bool operator<(const Expression& o) const;
  virtual bool operator<=(const Expression& o) const;
  virtual bool operator>(const Expression& o) const;
  virtual bool operator>=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator!=(const Expression& o) const;
};
}

inline biprog::Conditional::Conditional(boost::shared_ptr<Expression> cond,
    boost::shared_ptr<Expression> braces) :
    Conditioned(cond), Braced(braces) {
  //
}

inline biprog::Conditional::~Conditional() {
  //
}

inline bool biprog::Conditional::operator<(const Expression& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond < *expr.cond && *braces < *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Conditional::operator<=(const Expression& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond <= *expr.cond && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Conditional::operator>(const Expression& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond > *expr.cond && *braces > *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Conditional::operator>=(const Expression& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond >= *expr.cond && *braces >= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Conditional::operator==(const Expression& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond == *expr.cond && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Conditional::operator!=(const Expression& o) const {
  try {
    const Conditional& expr = dynamic_cast<const Conditional&>(o);
    return *cond != *expr.cond || *braces != *expr.braces;
  } catch (std::bad_cast e) {
    return true;
  }
}

#endif
