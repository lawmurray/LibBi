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
    bool bval;
	int ival;
	double rval;
	char *sval;
}

%token MODEL
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
%token <bval> BOOLEAN_LITERAL
%token <ival> INTEGER_LITERAL
%token <rval> REAL_LITERAL
%token <sval> STRING_LITERAL
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
%token IDENTIFIER
%token ENDL
%token OTHER

%%

/***************************************************************************
 * Expressions                                                             *
 ***************************************************************************/

type
    : IDENTIFIER  { type($1); }
    ;

real_literal
    : REAL_LITERAL  { real_literal($1); }
    ;
    
integer_literal
    : INTEGER_LITERAL  { integer_literal($1); }
    ;

boolean_literal
    : BOOLEAN_LITERAL  { boolean_literal($1); }
    ;

string_literal
    : STRING_LITERAL  { string_literal($1); }
    ;
    
body
    : '{' statements ';' '}'  { body($2); }
    | '{' statements '}'      { body($2); }
    | '{' '}'                 { body(); }
    ;

symbol
    : IDENTIFIER ':' type  { symbol($1, $3); }
    | IDENTIFIER           { symbol($1); }
    ;
    
reference
    : symbol tuple body    { reference($1, undef, $2, $3); }
    | symbol body          { reference($1, undef, undef, $2); }
    | symbol tuple         { reference($1, undef, $2); }
    | symbol square_tuple  { reference($1, $2); }
    | symbol               { reference($1); }
    ;

traversal
    : reference
    | traversal '.' reference  { binary_expression($1, '.', $3); }
    ;
    
postfix_expression
    : real_literal
    | integer_literal
    | boolean_literal
    | string_literal
    | traversal
    | tuple
    | body
    ;

defaulted_expression
    : postfix_expression
    | postfix_expression RIGHT_DOUBLE_ARROW defaulted_expression  { defaulted_expression($1, $3); }
    ;

unary_operator
    : '+'
    | '-'
    | '!'
    ;
    
unary_expression
    : defaulted_expression
    | unary_operator unary_expression  { unary_expression($1, $2); }
    ;

pow_operator
    : POW_OP
    | ELEM_POW_OP
    ;

pow_expression
    : unary_expression
    | pow_expression pow_operator unary_expression  { binary_expression($1, $2, $3); }
    ;

multiplicative_operator
    : '*'
    | ELEM_MUL_OP
    | '/'
    | ELEM_DIV_OP
    | '%'
    ;

multiplicative_expression
    : pow_expression
    | multiplicative_expression multiplicative_operator pow_expression  { binary_expression($1, $2, $3); }
    ;

additive_operator
    : '+'
    | ELEM_ADD_OP
    | '-'
    | ELEM_SUB_OP
    ;

additive_expression
    : multiplicative_expression
    | additive_expression additive_operator multiplicative_expression  { binary_expression($1, $2, $3); }
    ;

shift_operator
    : LEFT_OP
    | RIGHT_OP
    ;

shift_expression
    : additive_expression
    | shift_expression shift_operator additive_expression  { binary_expression($1, $2, $3); }
    ;

relational_operator
    : '<'
    | '>'
    | LE_OP
    | GE_OP
    ;
    
relational_expression
    : shift_expression
    | relational_expression relational_operator shift_expression  { binary_expression($1, $2, $3); }
    ;

equality_operator
    : '='
    | NE_OP
    | '~'
    ;

equality_expression
    : relational_expression
    | equality_expression equality_operator relational_expression  { binary_expression($1, $2, $3); }
    ;

logical_and_expression
    : equality_expression
    | logical_and_expression AND_OP equality_expression  { binary_expression($1, $2, $3); }
    ;

logical_or_expression
    : logical_and_expression
    | logical_or_expression OR_OP logical_and_expression  { binary_expression($1, $2, $3); }
    ;

conditional_expression
    : logical_or_expression
    /*| logical_or_expression '?' conditional_expression ':' logical_or_expression  { ternary_expression($1, $2, $3, $4, $5); }*/
    ;

assignment_expression
    : conditional_expression
    | conditional_expression LEFT_ARROW assignment_expression  { binary_expression($1, $2, $3); }
    ;
    
expression
    : assignment_expression
    ;    

/***************************************************************************
 * Statements                                                              *
 ***************************************************************************/
 
const_declaration
    : CONST IDENTIFIER '=' expression  { const_declaration($2, $4); }
    ;

dim_declaration
    : DIM IDENTIFIER tuple  { dim_declaration($2, $3); }
    ;

dim_arg
    : IDENTIFIER  { reference($1); }
    ;

dim_args
    : dim_args ',' dim_arg  { append($1, $3); }
    | dim_arg               { append($1); }
    ;
 
var_declaration
    : var_type IDENTIFIER '[' dim_args ']' tuple  { var_declaration($2, $4, $6); }
    | var_type IDENTIFIER '[' dim_args ']'        { var_declaration($2, $4); }
    | var_type IDENTIFIER tuple                   { var_declaration($2, undef, $3); }
    | var_type IDENTIFIER                         { var_declaration($2); }
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
    : statements ';' statement  { append($1, $3); }
    | statements reference      { append($1, $2); }
    | statement                 { append($1); }
    ;

/***************************************************************************
 * Tuples                                                                  *
 ***************************************************************************/

element
    : statements
    ;
    
elements
    : elements ',' element  { append($1, $3); }
    | element               { append($1); }
    ;
    
tuple
    : '(' elements ')'  { tuple($2); }
    | '(' ')'           { tuple(); }
    ;    

index
    : postfix_expression                                { index($1); }
    | postfix_expression DOUBLE_DOT postfix_expression  { index($1, $3); }
    ;

indexes
    : indexes ',' index  { append($1, $3); }
    | index              { append($1); }
    ;

square_tuple
    : '[' indexes ']'  { tuple($2); }
    ;      
    
/***************************************************************************
 * Files                                                                   *
 ***************************************************************************/

model
    : MODEL IDENTIFIER tuple body  { model($2, $3, $4); }
    | MODEL IDENTIFIER tuple ';'   { model($2, $3); }
    | MODEL IDENTIFIER body        { model($2, undef, $3); }
    | MODEL IDENTIFIER ';'         { model($2); }
    ;
    
method
    : METHOD IDENTIFIER tuple body  { method($2, $3, $4); }
    ;

function
    : FUNCTION IDENTIFIER tuple RIGHT_ARROW tuple body  { function($2, $3, $5, $6); }
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
