/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_STRINGLITERAL_HPP
#define BI_PROGRAM_STRINGLITERAL_HPP

#include "Literal.hpp"

#include <string>

namespace biprog {
/**
 * String literal.
 *
 * @ingroup program
 */
class StringLiteral: public Literal<std::string> {
public:
  /**
   * Constructor.
   */
  StringLiteral(const char* value);

  /**
   * Destructor.
   */
  virtual ~StringLiteral();
};
}

inline biprog::StringLiteral::StringLiteral(const char* value) :
    Literal<std::string>(value) {
  //
}

inline biprog::StringLiteral::~StringLiteral() {
  //
}

#endif
