GenBi
------------------------------------------------------------------------------

Code generator component of the Bi ("Bayesian inference") package.


Requirements
------------------------------------------------------------------------------

The following (non-standard) Perl modules are required:

  * Template,
  * Math::Symbolic,
  * Carp::Assert,
  * Parse::Yapp,
  * Parse::Lex,
  * File::Slurp,
  * Getopt::ArgvFile.

The following packages are optional for building the manual and visualising
models:

  * dot,
  * graphviz,
  * pod2latex,
  * latex2html,
  * pdflatex.


Installation
------------------------------------------------------------------------------

To install GenBi, simply place its files in any convenient location, and add
its bin/ directory to your PATH.


Documentation
------------------------------------------------------------------------------

Documentation of all GenBi Perl modules is available via perldoc, e.g.:

perldoc src/Bi/Model.pm

The user manual may be built, from the base directory of GenBi, using the
following:

bin/bi_make_user_docs

The HTML version of the manual may then be accessed at docs/html/index.htm,
and the LaTeX version built using the following sequence of commands from
within the docs/tex directory:

pdflatex index.tex
makeindex index
pdflatex index.tex
pdflatex index.tex

so as to produce the manual in the index.pdf file of that directory.
