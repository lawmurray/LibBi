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
class Conditional;
class Dim;
class EmptyExpression;
class Function;
class Group;
template<class T1> class Literal;
class Loop;
class Method;
class Model;
class Reference;
class Type;
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

  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<BinaryExpression> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Conditional> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Dim> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<EmptyExpression> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Function> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Group> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Literal<bool> > o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Literal<int> > o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Literal<double> > o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Literal<std::string> > o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Loop> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Method> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Model> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Reference> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Type> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<UnaryExpression> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Var> o) = 0;
};
}

#endif
