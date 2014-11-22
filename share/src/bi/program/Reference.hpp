/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_PROGRAM_REFERENCE_HPP
#define BI_PROGRAM_REFERENCE_HPP

#include "Named.hpp"
#include "Bracketed.hpp"
#include "Parenthesised.hpp"
#include "Typed.hpp"
#include "Braced.hpp"

namespace biprog {
/**
 * Binary expression.
 *
 * @ingroup program
 */
class Reference: public virtual Named,
    public virtual Bracketed,
    public virtual Parenthesised,
    public virtual Typed,
    public virtual Braced,
    public virtual boost::enable_shared_from_this<Reference> {
public:
  /**
   * Constructor.
   *
   * @param name Name.
   * @param brackets Expression in square brackets.
   * @param parens Expression in parentheses.
   * @param type Type.
   * @param braces Expression in braces.
   * @param target Target of the reference. May be null if unresolved.
   */
  Reference(const char* name, boost::shared_ptr<Typed> brackets,
      boost::shared_ptr<Typed> parens,
      boost::shared_ptr<Typed> type,
      boost::shared_ptr<Typed> braces,
      boost::shared_ptr<Typed> target);

  /**
   * Destructor.
   */
  virtual ~Reference();

  virtual boost::shared_ptr<Typed> accept(Visitor& v);

  virtual bool operator<=(const Typed& o) const;
  virtual bool operator==(const Typed& o) const;

protected:
  /**
   * Output.
   */
  virtual void output(std::ostream& out) const;

  /**
   * Target.
   */
  boost::shared_ptr<Typed> target;
};
}

inline biprog::Reference::Reference(const char* name,
    boost::shared_ptr<Typed> brackets,
    boost::shared_ptr<Typed> parens, boost::shared_ptr<Typed> type,
    boost::shared_ptr<Typed> braces,
    boost::shared_ptr<Typed> target) :
    Named(name), Bracketed(brackets), Parenthesised(parens), Typed(type), Braced(
        braces), target(target) {
  //
}

inline biprog::Reference::~Reference() {
  //
}

#endif

