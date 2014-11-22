/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_TYPED_HPP
#define BI_PROGRAM_TYPED_HPP

#include "Expression.hpp"

namespace biprog {
/**
 * Typed object.
 *
 * @ingroup program
 */
class Typed: public virtual Expression {
public:
  /**
   * Constructor.
   *
   * @param type type.
   */
  Typed();

  /**
   * Constructor.
   *
   * @param type type.
   */
  Typed(boost::shared_ptr<Typed> type);

  /**
   * Destructor.
   */
  virtual ~Typed() = 0;

  /**
   * Accept visitor.
   *
   * @param v The visitor.
   *
   * @return New expression with which to replace this one (may be the same).
   */
  virtual boost::shared_ptr<Typed> accept(Visitor& v) = 0;

  /**
   * Type.
   */
  boost::shared_ptr<Typed> type;
};
}

inline biprog::Typed::Typed() :
    type(boost::shared_ptr<Typed>()) {
  //
}

inline biprog::Typed::Typed(boost::shared_ptr<Typed> type) :
    type(type) {
  //
}

inline biprog::Typed::~Typed() {
  //
}

#endif
