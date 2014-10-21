%{
#define YY_DCL extern "C" int yylex()
%}

D                                  [0-9]
L                                  [a-zA-Z_]
H                                  [a-fA-F0-9]
E                                  [Ee][+-]?{D}+
FS                                 (f|F|l|L)
IS                                 (u|U|l|L)*

%x COMMENT_EOL COMMENT_INLINE

%%

[ \t]                               { count(); }
"//"                                { BEGIN(COMMENT_EOL); }
<COMMENT_EOL>"\n"                   { count(); BEGIN(INITIAL); }
"/*"                                { BEGIN(COMMENT_INLINE); }
<COMMENT_INLINE>"*/"                { BEGIN(INITIAL); }
<COMMENT_INLINE>"\n"                { count(); }
<COMMENT_EOL,COMMENT_INLINE>.       { count(); }

"model"                             { count(); return MODEL; }
"function"                          { count(); return FUNCTION; }
"method"                            { count(); return METHOD; }
"builtin"                           { count(); return BUILTIN; }
"const"                             { count(); return CONST; }
"dim"                               { count(); return DIM; }
"input"                             { count(); return INPUT; }
"hyper"                             { count(); return HYPER; }
"param"                             { count(); return PARAM; }
"state"                             { count(); return STATE; }
"obs"                               { count(); return OBS; }

{L}({L}|{D})*                       { count(); return IDENTIFIER; }                   

"true"                              { count(); return BOOLEAN_LITERAL; }
"false"                             { count(); return BOOLEAN_LITERAL; }
0[xX]{H}+{IS}?                      { count(); return INTEGER_LITERAL; }
0{D}+{IS}?                          { count(); return INTEGER_LITERAL; }
{D}+{IS}?                           { count(); return INTEGER_LITERAL; }
L?\'(\.|[^\'])+\'                   { count(); return INTEGER_LITERAL; }
{D}+{E}{FS}?                        { count(); return REAL_LITERAL; }
{D}*\.{D}+({E})?{FS}?               { count(); return REAL_LITERAL; }
{D}+\.{D}*({E})?{FS}?               { count(); return REAL_LITERAL; }
L?\"(\.|[^\"])*\"                   { count(); return STRING_LITERAL; }
L?\'(\.|[^\'])*\'                   { count(); return STRING_LITERAL; }

"->"                                { count(); return RIGHT_ARROW; }
"<-"                                { count(); return LEFT_ARROW; } 
"=>"                                { count(); return RIGHT_DOUBLE_ARROW; }
".."                                { count(); return DOUBLE_DOT; }
">>"                                { count(); return RIGHT_OP; }
"<<"                                { count(); return LEFT_OP; }
"&&"                                { count(); return AND_OP; }
"||"                                { count(); return OR_OP; }
"<="                                { count(); return LE_OP; }
">="                                { count(); return GE_OP; }
"=="                                { count(); return EQ_OP; }
"!="                                { count(); return NE_OP; }
"**"                                { count(); return POW_OP; }
".*"                                { count(); return ELEM_MUL_OP; }
"./"                                { count(); return ELEM_DIV_OP; }
".**"                               { count(); return ELEM_POW_OP; }
";"                                 { count(); return ';'; }
"{"                                 { count(); return '{'; }
"}"                                 { count(); return '}'; }
","                                 { count(); return ','; }
":"                                 { count(); return ':'; }
"="                                 { count(); return '='; }
"("                                 { count(); return '('; }
")"                                 { count(); return ')'; }
"["                                 { count(); return '['; }
"]"                                 { count(); return ']'; }
"."                                 { count(); return '.'; }
"&"                                 { count(); return '&'; }
"!"                                 { count(); return '!'; }
"~"                                 { count(); return '~'; }
"-"                                 { count(); return '-'; }
"+"                                 { count(); return '+'; }
"*"                                 { count(); return '*'; }
"/"                                 { count(); return '/'; }
"%"                                 { count(); return '%'; }
"<"                                 { count(); return '<'; }
">"                                 { count(); return '>'; }
"^"                                 { count(); return '^'; }
"|"                                 { count(); return '|'; }
"?"                                 { count(); return '?'; }
"\n"                                { count(); return ENDL; }

%%

int line = 0, col = 0;

yywrap() {
  return(1);
}

void count() {
  int i;

  for (i = 0; yytext[i] != '\0'; ++i) {
    if (yytext[i] == '\n') {
      ++line;
      col = 0;
    } else if (yytext[i] == '\t') {
      col += 8 - (col % 8);
    } else {
      ++col;
    }
  }
  ECHO;
}
