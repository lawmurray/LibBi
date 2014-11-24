/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_BRACES_HPP
#define BI_PROGRAM_BRACES_HPP

#include "Typed.hpp"
#include "Scoped.hpp"

namespace biprog {
/**
 * Braces.
 *
 * @ingroup program
 */
class Braces: public virtual Typed,
    public virtual Scoped,
    public virtual boost::enable_shared_from_this<Braces> {
public:
  /**
   * Destructor.
   */
  virtual ~Braces();

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

inline biprog::Braces::~Braces() {
  //
}

#endif
