/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_CONDITIONAL_HPP
#define BI_PROGRAM_CONDITIONAL_HPP

#include "Conditioned.hpp"
#include "Braced.hpp"
#include "Expression.hpp"

namespace biprog {
/**
 * Conditional.
 *
 * @ingroup program
 */
class Conditional: public Conditioned,
    public Braced,
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

#endif
