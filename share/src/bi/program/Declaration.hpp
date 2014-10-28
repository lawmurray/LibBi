/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_DECLARATION_HPP
#define BI_PROGRAM_DECLARATION_HPP

#include "Statement.hpp"
#include "Named.hpp"
#include "Bracketed.hpp"
#include "Parenthesised.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Declaration.
 *
 * @ingroup program
 */
class Declaration: public Statement,
    public Named,
    public Bracketed,
    public Parenthesised,
    public Braced {
public:
  /**
   * Constructor.
   */
  Declaration(const char* name, boost::shared_ptr<Expression> brackets,
      boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces);

  /**
   * Destructor.
   */
  virtual ~Declaration() = 0;

  /**
   * Match to reference if possible.
   */
  bool match(boost::shared_ptr<Expression> brackets,
      boost::shared_ptr<Expression> parens,
      boost::shared_ptr<Expression> braces, Match& match);
};
}

inline biprog::Declaration::Declaration(const char* name,
    boost::shared_ptr<Expression> brackets,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces) :
    Named(name), Bracketed(brackets), Parenthesised(parens), Braced(braces) {
  //
}

inline biprog::Declaration::~Declaration() {
  //
}

inline bool biprog::Declaration::match(boost::shared_ptr<Expression> brackets,
    boost::shared_ptr<Expression> parens,
    boost::shared_ptr<Expression> braces, Match& match) {
  bool result = true;
  result = result
      && (this->brackets && this->brackets->match(brackets, match)
          || !this->brackets && !brackets);
  result = result
      && (this->parens && this->parens->match(parens, match)
          || !this->parens && !parens);
  result = result
      && (this->braces && this->braces->match(braces, match)
          || !this->braces && !braces);

  return result;
}

#endif
