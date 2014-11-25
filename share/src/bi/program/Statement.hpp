/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_STATEMENT_HPP
#define BI_PROGRAM_STATEMENT_HPP

#include "../misc/assert.hpp"

namespace biprog {
class Visitor;

/**
 * Statement.
 *
 * @ingroup program
 */
class Statement {
public:
  /**
   * Destructor.
   */
  virtual ~Statement() = 0;

  /**
   * Clone.
   */
  virtual Statement* clone() = 0;

  /**
   * Accept visitor.
   *
   * @param v The visitor.
   *
   * @return New statement with which to replace this one (may be the same).
   */
  virtual Statement* accept(Visitor& v) = 0;

  /*
   * Bool cast to check for non-empty statement.
   */
  virtual operator bool() const;

  /*
   * Comparison operators for comparing statements in terms of
   * specialisation.
   *
   * The first two are the most commonly used, and so overridden by derived
   * classes. The remainder are expressed in terms of these.
   */
  virtual bool operator<=(const Statement& o) const = 0;
  virtual bool operator==(const Statement& o) const = 0;
  bool operator<(const Statement& o) const;
  bool operator>(const Statement& o) const;
  bool operator>=(const Statement& o) const;
  bool operator!=(const Statement& o) const;

  /**
   * Output operator. Defers to output() for polymorphism.
   */
  friend std::ostream& operator<<(std::ostream& out, const Statement& stmt) {
    stmt.output(out);
    return out;
  }

protected:
  /**
   * Output to stream.
   */
  virtual void output(std::ostream& out) const = 0;
};
}

inline biprog::Statement::~Statement() {
  //
}

biprog::Statement::operator bool() const {
  return true;
}

inline bool biprog::Statement::operator<(const Statement& o) const {
  return *this <= o && *this != o;
}

inline bool biprog::Statement::operator>(const Statement& o) const {
  return !(*this <= o);
}

inline bool biprog::Statement::operator>=(const Statement& o) const {
  return !(*this < o);
}

inline bool biprog::Statement::operator!=(const Statement& o) const {
  return !(*this == o);
}

#endif
