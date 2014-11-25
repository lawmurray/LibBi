/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_PROGRAM_HPP
#define BI_PROGRAM_PROGRAM_HPP

#include "Scoped.hpp"

namespace biprog {
/**
 * Program.
 *
 * @ingroup program
 */
class Program: public virtual Scoped {
public:
  /**
   * Destructor.
   */
  virtual ~Program();

protected:
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Program::~Program() {
  //
}

#endif
