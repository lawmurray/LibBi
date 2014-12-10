/**
 * @file
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au*
 * $Rev$
 * $Date$
 */
#ifndef BI_VISITOR_VISITOR_HPP
#define BI_VISITOR_VISITOR_HPP

namespace biprog {
class Statement;
class Expression;

class BinaryExpression;
class Braces;
class Brackets;
class Conditional;
class Def;
class Dim;
class EmptyExpression;
class EmptyStatement;
class Expression;
class ExpressionStatement;
template<class T1> class Literal;
class Loop;
class Parentheses;
class Placeholder;
class Reference;
class ReturnStatement;
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

  virtual Statement* visitStatement(Conditional* o) = 0;
  virtual Statement* visitStatement(Def* o) = 0;
  virtual Statement* visitStatement(Dim* o) = 0;
  virtual Statement* visitStatement(EmptyStatement* o) = 0;
  virtual Statement* visitStatement(ExpressionStatement* o) = 0;
  virtual Statement* visitStatement(Braces* o) = 0;
  virtual Statement* visitStatement(Loop* o) = 0;
  virtual Statement* visitStatement(Placeholder* o) = 0;
  virtual Statement* visitStatement(Reference* o) = 0;
  virtual Statement* visitStatement(ReturnStatement* o) = 0;
  virtual Statement* visitStatement(Sequence* o) = 0;
  virtual Statement* visitStatement(Var* o) = 0;

  virtual Expression* visitExpression(BinaryExpression* o) = 0;
  virtual Expression* visitExpression(EmptyExpression* o) = 0;
  virtual Expression* visitExpression(Braces* o) = 0;
  virtual Expression* visitExpression(Brackets* o) = 0;
  virtual Expression* visitExpression(Literal<bool>* o) = 0;
  virtual Expression* visitExpression(Literal<int>* o) = 0;
  virtual Expression* visitExpression(Literal<double>* o) = 0;
  virtual Expression* visitExpression(Literal<std::string>* o) = 0;
  virtual Expression* visitExpression(Parentheses* o) = 0;
  virtual Expression* visitExpression(Reference* o) = 0;
  virtual Expression* visitExpression(UnaryExpression* o) = 0;
};
}

#endif
