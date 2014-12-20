/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_EMPTYSTATEMENT_HPP
#define BI_PROGRAM_EMPTYSTATEMENT_HPP

#include "Statement.hpp"

namespace biprog {
/**
 * Empty statement.
 *
 * @ingroup program
 */
class EmptyStatement: public virtual Statement {
public:
  /**
   * Destructor.
   */
  virtual ~EmptyStatement();

  virtual EmptyStatement* clone();
  virtual Statement* acceptStatement(Visitor& v);

  virtual operator bool() const;

  virtual bool operator<=(const Statement& o) const;
  virtual bool operator==(const Statement& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::EmptyStatement::~EmptyStatement() {
  //
}

#endif
