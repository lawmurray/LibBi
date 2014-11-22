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

  /*
   * Comparison operators for comparing expressions in terms of
   * specialisation.
   *
   * The first two are the most commonly used, and so overridden by derived
   * classes. The remainder are expressed in terms of these.
   */
  virtual bool operator<=(const Typed& o) const = 0;
  virtual bool operator==(const Typed& o) const = 0;
  bool operator<(const Typed& o) const;
  bool operator>(const Typed& o) const;
  bool operator>=(const Typed& o) const;
  bool operator!=(const Typed& o) const;

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

inline bool biprog::Typed::operator<(const Typed& o) const {
  return *this <= o && *this != o;
}

inline bool biprog::Typed::operator>(const Typed& o) const {
  return !(*this <= o);
}

inline bool biprog::Typed::operator>=(const Typed& o) const {
  return !(*this < o);
}

inline bool biprog::Typed::operator!=(const Typed& o) const {
  return !(*this == o);
}

#endif
