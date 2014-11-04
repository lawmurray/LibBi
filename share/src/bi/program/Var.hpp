/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_VAR_HPP
#define BI_PROGRAM_VAR_HPP

#include "Named.hpp"
#include "Bracketed.hpp"
#include "Typed.hpp"

namespace biprog {
/**
 * Variable.
 *
 * @ingroup program
 */
class Var: public Named,
    public Bracketed,
    public Typed,
    public boost::enable_shared_from_this<Var> {
public:
  /**
   * Constructor.
   */
  Var(const char* name, boost::shared_ptr<Expression> brackets, Type* type);

  /**
   * Destructor.
   */
  virtual ~Var();

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

inline biprog::Var::Var(const char* name,
    boost::shared_ptr<Expression> parens, Type* type) :
    Named(name), Bracketed(brackets), Typed(type) {
  //
}

inline biprog::Var::~Var() {
  //
}

inline bool biprog::Var::operator<(const Expression& o) const {
  try {
    const Var& expr = dynamic_cast<const Var&>(o);
    return *brackets < *expr.brackets && *type < *expr.type;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Var::operator<=(const Expression& o) const {
  try {
    const Var& expr = dynamic_cast<const Var&>(o);
    return *brackets <= *expr.brackets && *type <= *expr.type;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Var::operator>(const Expression& o) const {
  try {
    const Var& expr = dynamic_cast<const Var&>(o);
    return *brackets > *expr.brackets && *type > *expr.type;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Var::operator>=(const Expression& o) const {
  try {
    const Var& expr = dynamic_cast<const Var&>(o);
    return *brackets >= *expr.brackets && *type >= *expr.type;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Var::operator==(const Expression& o) const {
  try {
    const Var& expr = dynamic_cast<const Var&>(o);
    return *brackets == *expr.brackets && *type == *expr.type;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Var::operator!=(const Expression& o) const {
  try {
    const Var& expr = dynamic_cast<const Var&>(o);
    return *brackets != *expr.brackets || *type != *expr.type;
  } catch (std::bad_cast e) {
    return true;
  }
}

#endif
