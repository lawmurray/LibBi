---
layout: page
title: "LibBi: Developers"
menu: Developers
permalink: developers.html
---

## Developers ##

### Building Documentation ###

A [developer guide]() is available.

The user manual is available from the [website](http://www.libbi.org), or can
be built, from the base directory of LibBi, using the command:

    perl MakeDocs.PL

You will need LaTeX installed. The manual will be built in PDF format at
`docs/pdf/index.pdf`.

The Perl components of LibBi are documented using POD ("Plain Old
Documentation"), as is conventional for Perl modules. The easiest way to
inspect this is on the command line, using, e.g.

    perldoc lib/Bi/Model.pm

A HTML build is not yet available.

The C++ components of LibBi are documented using
[Doxygen](http://www.doxygen.org). HTML documentation can be built by running
`doxygen` in the base directory of LibBi, and opening the
`docs/dev/html/index.html` file in your browser.
