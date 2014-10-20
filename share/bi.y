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
    std::string valString;
    
	bi::Const* valConst;
	bi::Dim* valDim;
	bi::Var* valVar;
	bi::Model* valModel;
	bi::Method* valMethod;
	bi::Function* valFunction;
	bi::Tuple* valTuple;
	bi::Index* valIndex;
	bi::OperatorReference* valOperatorReference;
	bi::Expression* expression;
}

%token MODEL
%token TYPE
%token FUNCTION
%token METHOD
%token BUILTIN
%token DIM
%token CONST
%token HYPER
%token PARAM
%token INPUT
%token STATE
%token OBS
%token <valBool> BOOLEAN_LITERAL
%token <valInt> INTEGER_LITERAL
%token <valReal> REAL_LITERAL
%token <valString> STRING_LITERAL
%token RIGHT_ARROW
%token LEFT_ARROW
%token RIGHT_DOUBLE_ARROW
%token DOUBLE_DOT
%token RIGHT_OP
%token LEFT_OP
%token AND_OP
%token OR_OP
%token LE_OP
%token GE_OP
%token EQ_OP
%token NE_OP
%token POW_OP
%token ELEM_MUL_OP
%token ELEM_DIV_OP
%token ELEM_ADD_OP
%token ELEM_SUB_OP
%token ELEM_POW_OP
%token COMMENT_EOL
%token COMMENT_START
%token COMMENT_END
%token OP
%token <sval> IDENTIFIER
%token ENDL
%token OTHER

%type <valConst> const_declaration
%type <valDim> dim_declaration
%type <valVar> var_declaration
%type <valModel> model
%type <valMethod> method
%type <valFunction> function
%type <valTuple> tuple
%type <valIndex> index
%type <valOperatorReference> traversal_operator unary_operator pow_operator multiplicative_operator additive_operator shift_operator relational_operator equality_operator
%type <valExpression> boolean_literal integer_literal real_literal string_literal symbol reference dim_arg traversal postfix_expression defaulted_expression unary_expression pow_expression multiplicative_expression additive_expression shift_expression relational_expression equality_expression logical_and_expression logical_or_expression conditional_expression assignment_expression expression 

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
    : '.'  {  return new bi::OperatorReference($1); }
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
    : postfix_expression
    | postfix_expression RIGHT_DOUBLE_ARROW defaulted_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

unary_operator
    : '+'  { return new bi::OperatorReference($1); }
    | '-'  { return new bi::OperatorReference($1); }
    | '!'  { return new bi::OperatorReference($1); }
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
    : '*'          { return new bi::OperatorReference($1); }
    | ELEM_MUL_OP  { return new bi::OperatorReference($1); }
    | '/'          { return new bi::OperatorReference($1); }
    | ELEM_DIV_OP  { return new bi::OperatorReference($1); }
    | '%'
    ;

multiplicative_expression
    : pow_expression
    | multiplicative_expression multiplicative_operator pow_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

additive_operator
    : '+'          { return new bi::OperatorReference($1); }
    | ELEM_ADD_OP  { return new bi::OperatorReference($1); }
    | '-'          { return new bi::OperatorReference($1); }
    | ELEM_SUB_OP  { return new bi::OperatorReference($1); }
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
    : '<'    { return new bi::OperatorReference($1); }
    | '>'    { return new bi::OperatorReference($1); }
    | LE_OP  { return new bi::OperatorReference($1); }
    | GE_OP  { return new bi::OperatorReference($1); }
    ;
    
relational_expression
    : shift_expression
    | relational_expression relational_operator shift_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

equality_operator
    : '='    { return new bi::OperatorReference($1); }
    | NE_OP  { return new bi::OperatorReference($1); }
    | '~'    { return new bi::OperatorReference($1); }
    ;

equality_expression
    : relational_expression
    | equality_expression equality_operator relational_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

logical_and_expression
    : equality_expression
    | logical_and_expression AND_OP equality_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

logical_or_expression
    : logical_and_expression
    | logical_or_expression OR_OP logical_and_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

conditional_expression
    : logical_or_expression
    /*| logical_or_expression '?' conditional_expression ':' logical_or_expression  { return new bi::TernaryOperator($1, $2, $3, $4, $5); }*/
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

dim_arg
    : IDENTIFIER  { return new bi::Reference($1); }
    ;

dim_args
    : dim_args ',' dim_arg  { return new bi::BinaryOperator($1, $2, $3); }
    | dim_arg               { return $1; }
    ;
 
var_declaration
    : var_type symbol '[' dim_args ']' tuple  { return new bi::Var($2, $4, $6); }
    | var_type symbol '[' dim_args ']'        { return new bi::Var($2, $4); }
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
    : statements ';' statement  { return new bi::BinaryOperator($1, $2, $3); }
    | statement ';'             { return $1; }
    ;

/***************************************************************************
 * Tuples                                                                  *
 ***************************************************************************/

element
    : statements
    ;
    
elements
    : elements ',' element  { return new bi::BinaryOperator($1, $2, $3); }
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
