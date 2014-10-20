%{
#define YY_DCL extern "C" int yylex()
%}

D                                  [0-9]
L                                  [a-zA-Z_]
H                                  [a-fA-F0-9]
E                                  [Ee][+-]?{D}+
FS                                 (f|F|l|L)
IS                                 (u|U|l|L)*

%x COMMENT_EOL
%x COMMENT_INLINE

%%

[ \t]                               { colinc(); }
"//"                                { BEGIN(COMMENT_EOL); }
<COMMENT_EOL>"\n"                   { lininc(); BEGIN(INITIAL); }
"/*"                                { BEGIN(COMMENT_INLINE); }
<COMMENT_INLINE>"*/"                { BEGIN(INITIAL); }
<COMMENT_INLINE>"\n"                { lininc(); }
<COMMENT_EOL,COMMENT_INLINE>.       { colinc(); }
"model"                             { colinc(); return MODEL; }
"function"                          { colinc(); return FUNCTION; }
"method"                            { colinc(); return METHOD; }
"builtin"                           { colinc(); return BUILTIN; }
"dim"                               { colinc(); return DIM; }
"const"                             { colinc(); return CONST; }
"input"                             { colinc(); return INPUT; }
"hyper"                             { colinc(); return HYPER; }
"param"                             { colinc(); return PARAM; }
"state"                             { colinc(); return STATE; }
"obs"                               { colinc(); return OBS; }
"true"                              { colinc(); return BOOLEAN_LITERAL; }
"false"                             { colinc(); return BOOLEAN_LITERAL; }
0[xX]{H}+{IS}?                      { colinc(); return INTEGER_LITERAL; }
0{D}+{IS}?                          { colinc(); return INTEGER_LITERAL; }
{D}+{IS}?                           { colinc(); return INTEGER_LITERAL; }
L?\'(\.|[^\'])+\'                   { colinc(); return INTEGER_LITERAL; }
{D}+{E}{FS}?                        { colinc(); return REAL_LITERAL; }
{D}*\.{D}+({E})?{FS}?               { colinc(); return REAL_LITERAL; }
{D}+\.{D}*({E})?{FS}?               { colinc(); return REAL_LITERAL; }
L?\"(\.|[^\"])*\"                   { colinc(); return STRING_LITERAL; }
L?\'(\.|[^\'])*\'                   { colinc(); return STRING_LITERAL; }
"->"                                { colinc(); return RIGHT_ARROW; }
"<-"                                { colinc(); return LEFT_ARROW; } 
"=>"                                { colinc(); return RIGHT_DOUBLE_ARROW; }
".."                                { colinc(); return DOUBLE_DOT; }
">>"                                { colinc(); return RIGHT_OP; }
"<<"                                { colinc(); return LEFT_OP; }
"&&"                                { colinc(); return AND_OP; }
"||"                                { colinc(); return OR_OP; }
"<="                                { colinc(); return LE_OP; }
">="                                { colinc(); return GE_OP; }
"=="                                { colinc(); return EQ_OP; }
"!="                                { colinc(); return NE_OP; }
"**"                                { colinc(); return POW_OP; }
".*"                                { colinc(); return ELEM_MUL_OP; }
"./"                                { colinc(); return ELEM_DIV_OP; }
".+"                                { colinc(); return ELEM_ADD_OP; }
".-"                                { colinc(); return ELEM_SUB_OP; }
".**"                               { colinc(); return ELEM_POW_OP; }
[;,\.:=()\[\]\{\}&!~\-+*\/%<>\^|?]  { colinc(); return OP; }  
{L}({L}|{D})*                       { colinc(); return IDENTIFIER; }                   
"\n"                                { lininc(); return ENDL; }

%%

int linno = 0, colno = 0;

yywrap() {
  return(1);
}

void lininc() {
  ++linno;
}

void colinc() {
  int i;

  for (i = 0; yytext[i] != '\0'; ++i) {
    if (yytext[i] == '\n') {
      colno = 0;
    } else if (yytext[i] == '\t') {
      colno += 8 - (colno % 8);
    } else {
      ++colno;
    }
  }
  ECHO;
}
