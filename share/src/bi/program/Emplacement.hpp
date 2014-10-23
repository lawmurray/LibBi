/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EMPLACEMENT_HPP
#define BI_PROGRAM_EMPLACEMENT_HPP

#include "Statement.hpp"
#include "Typed.hpp"
#include "Parenthesised.hpp"
#include "Bodied.hpp"

namespace biprog {
/**
 * Emplacement.
 *
 * @ingroup program
 */
class Emplacement: public Statement,
    public Typed,
    public Parenthesised,
    public Bodied {
public:
  /**
   * Constructor.
   */
  Emplacement(Type* type, Statement* in, Statement* body);

  /**
   * Destructor.
   */
  virtual ~Emplacement();
};
}

inline biprog::Emplacement::Emplacement(Type* type, Statement* in,
    Statement* body) :
    Typed(type), Parenthesised(in), Bodied(body) {
  //
}

inline biprog::Emplacement::~Emplacement() {
  //
}

#endif

