/**
 * Generator for C++ from LibBi code. 
 */
class GenCpp(hppFile:String, cppFile:String) {
  var hpp:TextFile(hppFile);
  var cpp:TextFile(cppFile);
  var indent:Int(0);  // current indent amount

  /**
   * Generate C++ for function.
   */
  method gen({ function f(args) -> (outs) { statements } }) {
    cpp.printin();
    gen(outs);
    cpp.print(' ');
    gen(f);
    cpp.print('(');
    gen(args);
    cpp.println(') {');
    indent <- indent + 1;
    gen(statements);
    indent <- indent - 1;
    cpp.printin(indent);
    cpp.println('}');
  }
}
