/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
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
  using Expression::operator<;
  using Expression::operator<=;
  using Expression::operator>;
  using Expression::operator>=;
  using Expression::operator==;
  using Expression::operator!=;
  virtual bool operator<(const Var& o) const;
  virtual bool operator<=(const Var& o) const;
  virtual bool operator>(const Var& o) const;
  virtual bool operator>=(const Var& o) const;
  virtual bool operator==(const Var& o) const;
  virtual bool operator!=(const Var& o) const;
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

inline bool biprog::Var::operator<(const Var& o) const {
  return *brackets < *o.brackets && *type < *o.type;
}

inline bool biprog::Var::operator<=(const Var& o) const {
  return *brackets <= *o.brackets && *type <= *o.type;
}

inline bool biprog::Var::operator>(const Var& o) const {
  return *brackets > *o.brackets && *type > *o.type;
}

inline bool biprog::Var::operator>=(const Var& o) const {
  return *brackets >= *o.brackets && *type >= *o.type;
}

inline bool biprog::Var::operator==(const Var& o) const {
  return *brackets == *o.brackets && *type == *o.type;
}

inline bool biprog::Var::operator!=(const Var& o) const {
  return *brackets != *o.brackets || *type != *o.type;
}

#endif
