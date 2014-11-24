/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 */
#ifndef BI_VISITOR_VISITOR_HPP
#define BI_VISITOR_VISITOR_HPP

#include "boost/shared_ptr.hpp"

namespace biprog {
class BinaryExpression;
class Braces;
class Brackets;
class Conditional;
class Def;
class Dim;
class EmptyExpression;
class Expression;
template<class T1> class Literal;
class Loop;
class Parentheses;
class Reference;
class Typed;
class ReturnExpression;
class Sequence;
class UnaryExpression;
class Var;

/**
 * Visitor.
 */
class Visitor {
public:
  /**
   * Destructor.
   */
  virtual ~Visitor();

  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<BinaryExpression> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Conditional> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Def> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Dim> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<EmptyExpression> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Braces> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Brackets> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Literal<bool> > o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Literal<int> > o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Literal<double> > o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Literal<std::string> > o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Loop> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Parentheses> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Reference> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<ReturnExpression> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Sequence> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<UnaryExpression> o) = 0;
  virtual boost::shared_ptr<Typed> visit(boost::shared_ptr<Var> o) = 0;
};
}

#endif
