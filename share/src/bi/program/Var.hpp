/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_VAR_HPP
#define BI_PROGRAM_VAR_HPP

#include "Named.hpp"
#include "Bracketed.hpp"
#include "Typed.hpp"

namespace biprog {
/**
 * Variable.
 *
 * @ingroup program
 */
class Var: public virtual Named,
    public virtual Bracketed,
    public virtual Typed,
    public virtual boost::enable_shared_from_this<Var> {
public:
  /**
   * Constructor.
   */
  Var(const char* name, boost::shared_ptr<Typed> brackets,
      boost::shared_ptr<Typed> type);

  /**
   * Destructor.
   */
  virtual ~Var();

  virtual boost::shared_ptr<Typed> accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;
};
}

inline biprog::Var::Var(const char* name,
    boost::shared_ptr<Typed> brackets,
    boost::shared_ptr<Typed> type) :
    Named(name), Bracketed(brackets), Typed(type) {
  //
}

inline biprog::Var::~Var() {
  //
}

#endif
