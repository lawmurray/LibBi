/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csirexpr.au>
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
  virtual bool operator<(const Expression& o) const;
  virtual bool operator<=(const Expression& o) const;
  virtual bool operator>(const Expression& o) const;
  virtual bool operator>=(const Expression& o) const;
  virtual bool operator==(const Expression& o) const;
  virtual bool operator!=(const Expression& o) const;
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

inline bool biprog::Reference::operator<(const Expression& o) const {
  try {
    const Reference& expr = dynamic_cast<const Reference&>(o);
    return *brackets < *expr.brackets && *parens < *expr.parens
        && *braces < *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Reference::operator<=(const Expression& o) const {
  try {
    const Reference& expr = dynamic_cast<const Reference&>(o);
    return *brackets <= *expr.brackets && *parens <= *expr.parens
        && *braces <= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Reference::operator>(const Expression& o) const {
  try {
    const Reference& expr = dynamic_cast<const Reference&>(o);
    return *brackets > *expr.brackets && *parens > *expr.parens
        && *braces > *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Reference::operator>=(const Expression& o) const {
  try {
    const Reference& expr = dynamic_cast<const Reference&>(o);
    return *brackets >= *expr.brackets && *parens >= *expr.parens
        && *braces >= *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Reference::operator==(const Expression& o) const {
  try {
    const Reference& expr = dynamic_cast<const Reference&>(o);
    return *brackets == *expr.brackets && *parens == *expr.parens
        && *braces == *expr.braces;
  } catch (std::bad_cast e) {
    return false;
  }
}

inline bool biprog::Reference::operator!=(const Expression& o) const {
  try {
    const Reference& expr = dynamic_cast<const Reference&>(o);
    return *brackets != *expr.brackets || *parens != *expr.parens
        || *braces != *expr.braces;
  } catch (std::bad_cast e) {
    return true;
  }
}

#endif

