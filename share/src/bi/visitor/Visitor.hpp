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

  virtual Typed* visit(BinaryExpression* o) = 0;
  virtual Typed* visit(Conditional* o) = 0;
  virtual Typed* visit(Def* o) = 0;
  virtual Typed* visit(Dim* o) = 0;
  virtual Typed* visit(EmptyExpression* o) = 0;
  virtual Typed* visit(Braces* o) = 0;
  virtual Typed* visit(Brackets* o) = 0;
  virtual Typed* visit(Literal<bool>* o) = 0;
  virtual Typed* visit(Literal<int>* o) = 0;
  virtual Typed* visit(Literal<double>* o) = 0;
  virtual Typed* visit(Literal<std::string>* o) = 0;
  virtual Typed* visit(Loop* o) = 0;
  virtual Typed* visit(Parentheses* o) = 0;
  virtual Typed* visit(Reference* o) = 0;
  virtual Typed* visit(ReturnExpression* o) = 0;
  virtual Typed* visit(Sequence* o) = 0;
  virtual Typed* visit(UnaryExpression* o) = 0;
  virtual Typed* visit(Var* o) = 0;
};
}

#endif
