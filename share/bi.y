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
    
    bi::OperatorReference* valOperatorReference;
    bi::Statement* valStatement;
    bi::Model* valModel;
    bi::Method* valMethod;
    bi::Function* valFunction;
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

%type <valOperatorReference> traversal_operator type_operator unary_operator pow_operator multiplicative_operator additive_operator shift_operator relational_operator equality_operator and_operator exclusive_or_operator inclusive_or_operator logical_and_operator logical_or_operator /*conditional_operator1 conditional_operator2*/ assignment_operator tuple_operator statement_operator
%type <valStatement> type boolean_literal integer_literal real_literal string_literal symbol reference traversal_expression type_expression postfix_expression defaulted_expression unary_expression pow_expression multiplicative_expression additive_expression shift_expression relational_expression equality_expression and_expression exclusive_or_expression inclusive_or_expression logical_and_expression logical_or_expression conditional_expression assignment_expression tuple_expression expression const_declaration dim_declaration var_declaration declaration statement
%type <valModel> model
%type <valMethod> method
%type <valFunction> function

%start file
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
    : IDENTIFIER  { return new bi::Symbol($1); }
    ;
    
reference
    : symbol '[' statement ']'                     { return new bi::Reference($1, $3); }
    | symbol '(' statement ')'                     { return new bi::Reference($1, NULL, $3); }
    | symbol '(' statement ')' '{' statement '}'   { return new bi::Reference($1, NULL, $3, $6); }
    | symbol '{' statement '}'                     { return new bi::Reference($1, NULL, NULL, $3); }
    | symbol                                       { return new bi::Reference($1); }
    ;

traversal_operator
    : '.'  { return new bi::OperatorReference('.'); }
    ;
    
traversal_expression
    : reference                                          { return $1; }
    | traversal_expression traversal_operator reference  { return new BinaryOperator($1, $2, $3); }
    ;
    
type_operator
    : ':'  { return new bi::OperatorReference(':'); }
    ;
    
type_expression
    : traversal_expression
    | traversal_expression type_operator type  { return new BinaryOperator($1, $2, $3); }
    ;
    
postfix_expression
    : type_expression
    | boolean_literal
    | integer_literal
    | real_literal
    | string_literal
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

and_operator
    : '&'  { return new bi::OperatorReference('&'); }
    ;
    
and_expression
	: equality_expression                              { return $1; }
	| and_expression and_operator equality_expression  { return new bi::BinaryOperator($1, $2, $3); }
	;

exclusive_or_operator
    : '^'  { return new bi::OperatorReference('^'); }
    ;

exclusive_or_expression
	: and_expression                                                { return $1; }
	| exclusive_or_expression exclusive_or_operator and_expression  { return new bi::BinaryOperator($1, $2, $3); }
	;

inclusive_or_operator
    : '|'  { return new bi::OperatorReference('|'); }
    ;

inclusive_or_expression
	: exclusive_or_expression                                                { return $1; }
	| inclusive_or_expression inclusive_or_operator exclusive_or_expression  { return new bi::BinaryOperator($1, $2, $3); }

logical_and_operator
    : AND_OP  { return new bi::OperatorReference($1); }
    ;

logical_and_expression
    : inclusive_or_expression                                          { return $1; }
    | logical_and_expression logical_and_operator equality_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

logical_or_operator
    : OR_OP  { return new bi::OperatorReference($1); }
    ;

logical_or_expression
    : logical_and_expression                                            { return $1; }
    | logical_or_expression logical_or_operator logical_and_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

/*
conditional_operator1
    : '?'  { return new bi::OperatorReference('?'); }
    ;

conditional_operator2
    : ':'  { return new bi::OperatorReference(':'); }
    ;
*/

conditional_expression
    : logical_or_expression
    /*| logical_or_expression conditional_operator1 conditional_expression conditional_operator2 logical_or_expression  { return new bi::TernaryOperator($1, $2, $3, $4, $5); }*/
    ;

assignment_operator
    : LEFT_ARROW  { return new bi::OperatorReference($1); }
    | '~'         { return new bi::OperatorReference('~'); }
    ;

assignment_expression
    : conditional_expression
    | conditional_expression assignment_operator assignment_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

tuple_operator
    : ','  { return new bi::OperatorReference(','); }
    ;
    
tuple_expression
    : assignment_expression
    | tuple_expression tuple_operator assignment_expression  { return new bi::BinaryOperator($1, $2, $3); }
    ;

expression
    : tuple_expression
    ;    


/***************************************************************************
 * Declarations                                                            *
 ***************************************************************************/
 
const_declaration
    : CONST reference  { return new bi::Const($2); }
    ;

dim_declaration
    : DIM reference  { return new bi::Dim($2); }
    ;

var_declaration
    : INPUT reference  { return new bi::Input($2); }
    | HYPER reference  { return new bi::Hyper($2); }
    | PARAM reference  { return new bi::Param($2); }
    | STATE reference  { return new bi::State($2); }
    | OBS reference  { return new bi::Obs($2); }
    ;

declaration
    : const_declaration
    | dim_declaration
    | var_declaration
    ;


/***************************************************************************
 * Statements                                                              *
 ***************************************************************************/

statement_operator
    : ';'  { return new bi::OperatorReference(';'); }
    ;

statement
    : expression
    | expression statement_operator
    | declaration
    | declaration statement_operator
    | expression statement_operator statement  { return new bi::BinaryOperator($1, $2, $3); }
    ;

    
/***************************************************************************
 * Files                                                                   *
 ***************************************************************************/

model
    : MODEL reference  { return new bi::Model($2); }
    ;
    
method
    : METHOD reference  { return new bi::Method($2); }
    ;

function
    : FUNCTION symbol '(' statement ')' RIGHT_ARROW '(' statement ')' '{' statement '}'  { return new bi::Function($2, $4, $8, $11); }
    | FUNCTION symbol '(' statement ')' RIGHT_ARROW '(' statement ')' '{' '}'            { return new bi::Function($2, $4, $8); }
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
