/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_DIM_HPP
#define BI_PROGRAM_DIM_HPP

#include "Named.hpp"
#include "Bracketed.hpp"

namespace biprog {
/**
 * Dimension.
 *
 * @ingroup program
 */
class Dim: public virtual Named,
    public virtual Bracketed,
    public boost::enable_shared_from_this<Dim> {
public:
  /**
   * Constructor.
   */
  Dim(const char* name, boost::shared_ptr<Expression> brackets);

  /**
   * Destructor.
   */
  virtual ~Dim();

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

inline biprog::Dim::Dim(const char* name,
    boost::shared_ptr<Expression> brackets) :
    Named(name), Bracketed(brackets) {
  //
}

inline biprog::Dim::~Dim() {
  //
}

inline bool biprog::Dim::operator<(const Expression& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets < *expr.brackets;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Dim::operator<=(const Expression& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets <= *expr.brackets;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Dim::operator>(const Expression& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets > *expr.brackets;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Dim::operator>=(const Expression& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets >= *expr.brackets;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Dim::operator==(const Expression& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets == *expr.brackets;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Dim::operator!=(const Expression& o) const {
  try {
    const Dim& expr = dynamic_cast<const Dim&>(o);
    return *brackets != *expr.brackets;
  } catch (std::bad_cast e) {
    return true;
  }
}

#endif
