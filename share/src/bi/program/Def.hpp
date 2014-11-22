/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_DEF_HPP
#define BI_PROGRAM_DEF_HPP

#include "Named.hpp"
#include "Parenthesised.hpp"
#include "Typed.hpp"
#include "Braced.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * Def.
 *
 * @ingroup program
 */
class Def: public virtual Named,
    public virtual Parenthesised,
    public virtual Typed,
    public virtual Braced,
    public virtual Scoped,
    public virtual boost::enable_shared_from_this<Def> {
public:
  /**
   * Constructor.
   */
  Def(const char* name, boost::shared_ptr<Typed> parens,
      boost::shared_ptr<Typed> type, boost::shared_ptr<Typed> braces,
      boost::shared_ptr<Scope> scope);

  /**
   * Destructor.
   */
  virtual ~Def();

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

inline biprog::Def::~Def() {
  //
}

#endif
