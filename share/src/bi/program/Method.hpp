/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_METHOD_HPP
#define BI_PROGRAM_METHOD_HPP

#include "Overloaded.hpp"

namespace biprog {
/**
 * Method.
 *
 * @ingroup program
 */
class Method: public virtual Overloaded,
    public virtual boost::enable_shared_from_this<Method> {
public:
  /**
   * Destructor.
   */
  virtual ~Method();

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Method::~Method() {
  //
}

#endif
