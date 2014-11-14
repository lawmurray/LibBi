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
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Braced> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Bracketed> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Conditional> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Dim> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<EmptyExpression> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Function> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<FunctionOverload> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Group> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Literal> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Loop> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Method> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<MethodOverload> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Model> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Operator> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Program> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Reference> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Sequence> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Type> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<UnaryExpression> o) = 0;
  virtual boost::shared_ptr<Expression> visit(boost::shared_ptr<Var> o) = 0;
};
}

#endif
