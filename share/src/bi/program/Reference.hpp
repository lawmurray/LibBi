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
class Reference: public virtual Named,
    public virtual Bracketed,
    public virtual Parenthesised,
    public virtual Braced,
    public virtual boost::enable_shared_from_this<Reference> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param brackets Expression in square brackets.
   * @param parens Expression in parentheses.
   * @param braces Expression in braces.
   * @param target Target of the reference. May be null if unresolved.
   */
  Reference(const char* name, boost::shared_ptr<Expression> brackets,
      boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces,
      boost::shared_ptr<Expression> target);

  /**
   * Destructor.
   */
  virtual ~Reference();

  virtual boost::shared_ptr<Expression> accept(Visitor& v);

    virtual bool operator<=(const Expression& o) const;
 virtual bool operator==(const Expression& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;

  /**
   * Target.
   */
  boost::shared_ptr<Expression> target;
};
}

inline biprog::Reference::Reference(const char* name,
    boost::shared_ptr<Expression> brackets,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces,
    boost::shared_ptr<Expression> target) :
    Named(name), Bracketed(brackets), Parenthesised(parens), Braced(braces), target(
        target) {
  //
}

inline biprog::Reference::~Reference() {
  //
}

#endif

