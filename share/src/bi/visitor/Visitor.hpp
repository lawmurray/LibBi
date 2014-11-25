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
template<class T1> class Literal;
class Loop;
class Parentheses;
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

  virtual Expression* visit(BinaryExpression* o) = 0;
  virtual Statement* visit(Conditional* o) = 0;
  virtual Statement* visit(Def* o) = 0;
  virtual Statement* visit(Dim* o) = 0;
  virtual Expression* visit(EmptyExpression* o) = 0;
  virtual Statement* visit(EmptyStatement* o) = 0;
  virtual Braces* visit(Braces* o) = 0;
  virtual Expression* visit(Brackets* o) = 0;
  virtual Expression* visit(Literal<bool>* o) = 0;
  virtual Expression* visit(Literal<int>* o) = 0;
  virtual Expression* visit(Literal<double>* o) = 0;
  virtual Expression* visit(Literal<std::string>* o) = 0;
  virtual Statement* visit(Loop* o) = 0;
  virtual Expression* visit(Parentheses* o) = 0;
  virtual Expression* visit(Reference* o) = 0;
  virtual Statement* visit(ReturnStatement* o) = 0;
  virtual Statement* visit(Sequence* o) = 0;
  virtual Expression* visit(UnaryExpression* o) = 0;
  virtual Statement* visit(Var* o) = 0;
};
}

#endif
