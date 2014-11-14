/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_FUNCTION_HPP
#define BI_PROGRAM_FUNCTION_HPP

#include "Overloaded.hpp"

namespace biprog {
/**
 * Function.
 *
 * @ingroup program
 */
class Function: public virtual Overloaded,
    public virtual boost::enable_shared_from_this<Function> {
public:
  /**
   * Destructor.
   */
  virtual ~Function();

  virtual boost::shared_ptr<Expression> accept(Visitor& v);

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Function::~Function() {
  //
}

#endif
