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

namespace biprog {
/**
 * Declaration.
 *
 * @ingroup program
 */
class Declaration: public Statement {
public:
  /**
   * Constructor.
   */
  Declaration();

  /**
   * Destructor.
   */
  virtual ~Declaration() = 0;
};
}

#endif
