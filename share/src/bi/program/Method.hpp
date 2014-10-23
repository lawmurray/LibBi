/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_METHOD_HPP
#define BI_PROGRAM_METHOD_HPP

#include "Declaration.hpp"
#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Bodied.hpp"

namespace biprog {
/**
 * Method.
 *
 * @ingroup program
 */
class Method: public Declaration,
    public Named,
    public Parenthesised,
    public Bodied {
public:
  /**
   * Constructor.
   */
  Method(const char* name, Statement* in = NULL, Statement* body = NULL);

  /**
   * Destructor.
   */
  virtual ~Method();
};
}

inline biprog::Method::Method(const char* name, Statement* in,
    Statement* body) :
    Named(name), Parenthesised(in), Bodied(body) {
  //
}

inline biprog::Method::~Method() {
  //
}

#endif

