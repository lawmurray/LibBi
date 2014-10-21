%{
#include <cstdio>
#include <iostream>

extern "C" int yylex();
extern "C" int yyparse();
extern "C" FILE *yyin;

extern int colno;
extern int linno;

void yyerror(const char *s);
%}

%union {
    bool valBool;
    int valInt;
    real valReal;
    char* valString;
    
	bi::Model* valModel;
	bi::Function* valFunction;
	bi::Method* valMethod;

	bi::Const* valConst;
	bi::Dim* valDim;
	bi::Var* valVar;
	bi::OperatorReference* valOperatorReference;
	bi::Expression* expression;
}

%token COMMENT_EOL COMMENT_START COMMENT_END
%token MODEL FUNCTION METHOD BUILTIN CONST DIM HYPER PARAM INPUT STATE OBS
%token <valString> IDENTIFIER
%token <valBool> BOOLEAN_LITERAL
%token <valInt> INTEGER_LITERAL
%token <valReal> REAL_LITERAL
%token <valString> STRING_LITERAL
%token <valString> RIGHT_ARROW LEFT_ARROW RIGHT_DOUBLE_ARROW DOUBLE_DOT
%token <valString> RIGHT_OP LEFT_OP AND_OP OR_OP LE_OP GE_OP EQ_OP NE_OP
%token <valString> POW_OP ELEM_MUL_OP ELEM_DIV_OP ELEM_POW_OP
%token ENDL
%token OTHER

%type <valConst> const_declaration
%type <valDim> dim_declaration
%type <valVar> var_declaration
%type <valModel> model
%type <valMethod> method
%type <valFunction> function
%type <valOperatorReference> traversal_operator unary_operator pow_operator multiplicative_operator additive_operator shift_operator relational_operator equality_operator
%type <valExpression> type boolean_literal integer_literal real_literal string_literal symbol reference dim_reference dim_references traversal postfix_expression defaulted_expression unary_expression pow_expression multiplicative_expression additive_expression shift_expression relational_expression equality_expression and_expression exclusive_or_expression inclusive_or_expression logical_and_expression logical_or_expression conditional_expression assignment_expression expression tuple square_tuple index statements 

%%

/***************************************************************************
 * Expressions                                                             *
 ***************************************************************************/

type
    : IDENTIFIER  { return new bi::Type($1); }
    ;

boolean_literal
    : BOOLEAN_LITERAL  { return new bi::BooleanLiteral($1); }
    ;

integer_literal
    : INTEGER_LITERAL  { return new bi::IntegerLiteral($1); }
    ;

real_literal
    : REAL_LITERAL  { return new bi::RealLiteral($1); }
    ;

string_literal
    : STRING_LITERAL  { return new bi::StringLiteral($1); }
    ;

symbol
    : IDENTIFIER ':' type  { return new bi::Symbol($1, $3); }
    | IDENTIFIER           { return new bi::Symbol($1); }
    ;
    
reference
    : symbol tuple '{' statements '}'  { return new bi::Reference($1, NULL, $2, $4); }
    | symbol '{' statements '}'        { return new bi::Reference($1, NULL, NULL, $3); }
    | symbol tuple                     { return new bi::Reference($1, NULL, $2); }
    | symbol square_tuple              { return new bi::Reference($1, $2); }
    | symbol                           { return new bi::Reference($1); }
    ;

traversal_operator
    : '.'  { return new bi::OperatorReference('.'); }
    ;
    
traversal
    : traversal traversal_operator reference  { return new BinaryOperator($1, $2, $3); }
    | reference                               { return $1; }
    ;
    
postfix_expression
    : boolean_literal
    | integer_literal
    | real_literal
    | string_literal
    | traversal
    | tuple
    ;

defaulted_expression
    : postfix_expression                                          { return $1; }
    | postfix_expression RIGHT_DOUBLE_ARROW defaulted_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

unary_operator
    : '+'  { return new bi::OperatorReference('+'); }
    | '-'  { return new bi::OperatorReference('-'); }
    | '!'  { return new bi::OperatorReference('!'); }
    ;
    
unary_expression
    : defaulted_expression
    | unary_operator unary_expression  { return new bi::UnaryOperator($1, $2); }
    ;

pow_operator
    : POW_OP       { return new bi::OperatorReference($1); }
    | ELEM_POW_OP  { return new bi::OperatorReference($1); }
    ;

pow_expression
    : unary_expression
    | pow_expression pow_operator unary_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

multiplicative_operator
    : '*'          { return new bi::OperatorReference('*'); }
    | ELEM_MUL_OP  { return new bi::OperatorReference($1); }
    | '/'          { return new bi::OperatorReference('/'); }
    | ELEM_DIV_OP  { return new bi::OperatorReference($1); }
    | '%'          { return new bi::OperatorReference('%'); }
    ;

multiplicative_expression
    : pow_expression
    | multiplicative_expression multiplicative_operator pow_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

additive_operator
    : '+'          { return new bi::OperatorReference('+'); }
    | '-'          { return new bi::OperatorReference('-'); }
    ;

additive_expression
    : multiplicative_expression
    | additive_expression additive_operator multiplicative_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

shift_operator
    : LEFT_OP   { return new bi::OperatorReference($1); }
    | RIGHT_OP  { return new bi::OperatorReference($1); }
    ;

shift_expression
    : additive_expression
    | shift_expression shift_operator additive_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

relational_operator
    : '<'    { return new bi::OperatorReference('<'); }
    | '>'    { return new bi::OperatorReference('>'); }
    | LE_OP  { return new bi::OperatorReference($1); }
    | GE_OP  { return new bi::OperatorReference($1); }
    ;
    
relational_expression
    : shift_expression
    | relational_expression relational_operator shift_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

equality_operator
    : EQ_OP  { return new bi::OperatorReference($1); }
    | NE_OP  { return new bi::OperatorReference($1); }
    | '~'    { return new bi::OperatorReference('~'); }
    ;

equality_expression
    : relational_expression                                        { return $1; }
    | equality_expression equality_operator relational_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

and_expression
	: equality_expression                     { return $1; }
	| and_expression '&' equality_expression  { return new bi::BinaryOperator($1, '&', $3); }
	;

exclusive_or_expression
	: and_expression                              { return $1; }
	| exclusive_or_expression '^' and_expression  { return new bi::BinaryOperator($1, '^', $3); }
	;

inclusive_or_expression
	: exclusive_or_expression                              { return $1; }
	| inclusive_or_expression '|' exclusive_or_expression  { return new bi::BinaryOperator($1, '|', $3); }

logical_and_expression
    : inclusive_or_expression                            { return $1; }
    | logical_and_expression AND_OP equality_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

logical_or_expression
    : logical_and_expression                              { return $1; }
    | logical_or_expression OR_OP logical_and_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

conditional_expression
    : logical_or_expression
    /*| logical_or_expression '?' conditional_expression ':' logical_or_expression  { return new bi::TernaryOperator($1, '?', $3, ':', $5); }*/
    ;

assignment_expression
    : conditional_expression
    | conditional_expression LEFT_ARROW assignment_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;
    
expression
    : assignment_expression
    ;    

/***************************************************************************
 * Statements                                                              *
 ***************************************************************************/
 
const_declaration
    : CONST IDENTIFIER '=' expression  { return new bi::Const($2, $4); }
    ;

dim_declaration
    : DIM IDENTIFIER tuple  { return new bi::Dim($2, $3); }
    ;

dim_reference
    : IDENTIFIER  { return new bi::DimReference($1); }
    ;

dim_references
    : dim_references ',' dim_reference  { return new bi::BinaryOperator($1, ',', $3); }
    | dim_reference                     { return $1; }
    ;
 
var_declaration
    : var_type symbol '[' dim_references ']' tuple  { return new bi::Var($2, $4, $6); }
    | var_type symbol '[' dim_references ']'        { return new bi::Var($2, $4); }
    | var_type symbol tuple                   { return new bi::Var($2, NULL, $3); }
    | var_type symbol                         { return new bi::Var($2); }
    ;

var_type
    : INPUT
    | HYPER
    | PARAM
    | STATE
    | OBS
    ;

declaration
    : const_declaration
    | dim_declaration
    | var_declaration
    ;

statement
    : declaration
    | expression
    ;
    
statements
    : statements ';' statement  { return new bi::BinaryOperator($1, ';', $3); }
    | statement ';'             { return ';'; }
    ;

/***************************************************************************
 * Tuples                                                                  *
 ***************************************************************************/

element
    : statements
    ;
    
elements
    : elements ',' element  { return new bi::BinaryOperator($1, ',', $3); }
    | element               { return $1; }
    ;
    
tuple
    : '(' elements ')'  { return new bi::Tuple($2); }
    | '(' ')'           { return new bi::Tuple(NULL); }
    ;    

index
    : postfix_expression                                { return new bi::Index($1); }
    | postfix_expression DOUBLE_DOT postfix_expression  { return new bi::Index($1, $3); }
    ;

indexes
    : indexes ',' index  { return new bi::BinaryOperator($1, $3); }
    | index              { return $1; }
    ;

square_tuple
    : '[' indexes ']'  { return new bi::Tuple($2); }
    | '[' ']'          { return new bi::Tuple(NULL); }
    ;      
    
/***************************************************************************
 * Files                                                                   *
 ***************************************************************************/

model
    : MODEL IDENTIFIER tuple '{' statements '}'  { return new bi::Model($2, $3, $5); }
    | MODEL IDENTIFIER '{' statements '}'        { return new bi::Model($2, NULL, $4); }
    ;
    
method
    : METHOD IDENTIFIER tuple '{' statements '}'  { return new bi::Method($2, $3, $5); }
    ;

function
    : FUNCTION IDENTIFIER tuple RIGHT_ARROW tuple '{' statements '}'  { return new bi::Function($2, $3, $5, $7); }
    ;

top
    : model
    | method
    | function
    ;

file
    : file top
    | top
    ;

%%

int main() {
  do {
    yyparse();
  } while (!feof(yyin));

  return 0;
}

void yyerror(const char *msg) {
  std::cerr << "Error (line " << linno << " col " << colno << "): " << msg << std::endl;
  exit(-1);
}
