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

namespace biprog {
/**
 * Declaration.
 *
 * @ingroup program
 */
class Declaration: public Statement, public Named {
public:
  /**
   * Constructor.
   */
  Declaration(const char* name);

  /**
   * Destructor.
   */
  virtual ~Declaration() = 0;
};
}

inline biprog::Declaration::Declaration(const char* name) : Named(name) {
  //
}

inline biprog::Declaration::~Declaration() {
  //
}

#endif
