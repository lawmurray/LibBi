---
layout: page
title: "LibBi: Getting started"
menu: Getting started
permalink: getting-started.html
---

## Getting started

### Download

Use the button on the left to download the latest version. The download is the
same for all platforms.

### Install

LibBi runs on Linux, and on Mac OS X with dependencies installed from the
[MacPorts](http://www.macports.org) project. It has not been tried on Windows;
[Cygwin](http://www.cygwin.com) would be a good bet, however, and if you do
get it running there the contribution of installation instructions would be
appreciated.

Installation instructions are in the `INSTALL` file of the distribution.

LibBi itself simply installs as a Perl module. The trick, however, is that the
C++ code it generates to run your models has a number of dependencies. If
these are not installed you will get errors much like this the first time you
run LibBi:

> Error: ./configure failed with return code 1, see
> .Model/build_assert/configure.log and .Model/build_assert/config.log for
> details

You should bring up the `configure.log` file mentioned. This is the output of
the usual `configure` script run when building software with a GNU Autotools
build system. You will be able to determine the missing dependency from there.

### Running

The [introductory paper](/papers) is a good place to start, followed by the
[user manual](/documentation). The examples in the introductory paper are
[available](/examples), along with others, for you to download and run.
