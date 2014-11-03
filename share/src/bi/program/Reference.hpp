/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_REFERENCE_HPP
#define BI_PROGRAM_REFERENCE_HPP

#include "Named.hpp"
#include "Bracketed.hpp"
#include "Parenthesised.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Binary expression.
 *
 * @ingroup program
 */
class Reference: public Named,
    public Bracketed,
    public Parenthesised,
    public Braced,
    public boost::enable_shared_from_this<Reference> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param brackets Expression in square brackets.
   * @param parens Expression in parentheses.
   * @param braces Expression in braces.
   */
  Reference(const char* name, boost::shared_ptr<Expression> brackets,
      boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~Reference();

  /*
   * Operators.
   */
  using Expression::operator<;
  using Expression::operator<=;
  using Expression::operator>;
  using Expression::operator>=;
  using Expression::operator==;
  using Expression::operator!=;
  virtual bool operator<(const Reference& o) const;
  virtual bool operator<=(const Reference& o) const;
  virtual bool operator>(const Reference& o) const;
  virtual bool operator>=(const Reference& o) const;
  virtual bool operator==(const Reference& o) const;
  virtual bool operator!=(const Reference& o) const;
};
}

inline biprog::Reference::Reference(const char* name,
    boost::shared_ptr<Expression> brackets,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces) :
    Named(name), Bracketed(brackets), Parenthesised(parens), Braced(braces) {
  //
}

inline biprog::Reference::~Reference() {
  //
}

inline bool biprog::Reference::operator<(const Reference& o) const {
  return *brackets < *o.brackets && *parens < *o.parens && *braces < *o.braces;
}

inline bool biprog::Reference::operator<=(const Reference& o) const {
  return *brackets <= *o.brackets && *parens <= *o.parens
      && *braces <= *o.braces;
}

inline bool biprog::Reference::operator>(const Reference& o) const {
  return *brackets > *o.brackets && *parens > *o.parens && *braces > *o.braces;
}

inline bool biprog::Reference::operator>=(const Reference& o) const {
  return *brackets >= *o.brackets && *parens >= *o.parens
      && *braces >= *o.braces;
}

inline bool biprog::Reference::operator==(const Reference& o) const {
  return *brackets == *o.brackets && *parens == *o.parens
      && *braces == *o.braces;
}

inline bool biprog::Reference::operator!=(const Reference& o) const {
  return *brackets != *o.brackets || *parens != *o.parens
      || *braces != *o.braces;
}

#endif

