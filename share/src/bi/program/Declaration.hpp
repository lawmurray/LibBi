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
#include "Reference.hpp"

#include "boost/shared_ptr.hpp"

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
  Declaration(Reference* ref);

  /**
   * Destructor.
   */
  virtual ~Declaration() = 0;

  /**
   * Reference.
   */
  boost::shared_ptr<Reference> ref;
};
}

inline biprog::Declaration::Declaration(Reference* ref) : ref(ref) {
  //
}

inline biprog::Declaration::~Declaration() {
  //
}

#endif
