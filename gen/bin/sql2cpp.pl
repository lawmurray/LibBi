#!/usr/bin/env perl

##
## Convert SQLite model specification to C++ code for bi.
##
## @author Lawrence Murray <lawrence.murray@csiro.au>
## $Rev: 587 $
## $Date: 2010-02-05 13:50:48 +0800 (Fri, 05 Feb 2010) $
##

use strict;
use DBI;
use DBD::SQLite;
use Pod::Usage;
use Carp;
use Getopt::Long;
use Template;
use Template::Filters;
use FindBin qw($Bin);

# command line arguments
my $model = '';
my $outdir = '.';
my $ttdir = "$Bin/../tt/cpp";
Getopt::Long::Configure("auto_help");
GetOptions("model=s" => \$model,
    "outdir=s" => \$outdir,
    "ttdir=s" => \$ttdir);

$model =~ /(\w+)\.db/i;
my $name = ucfirst($1);

# connect to database
$ENV{'DBI_DSN'} = "dbi:SQLite:dbname=$model";
my $dbh = DBI->connect($ENV{'DBI_DSN'});
my $sth = $dbh->prepare("SELECT Name, HasX, HasY, HasZ FROM Node " .
    "WHERE Category <> 'Intermediate result' AND Category <> 'Constant'")
    || die();

# template toolkit setup
my $tt = Template->new({
    INCLUDE_PATH => $ttdir,
    FILTERS => {
      'cudaexp' => \&cudaexp,
      'latexexp' => \&latexexp
    }
}) || die "$Template::ERROR\n";
my $vars = {
  'Name' => $name,
  'Model' => $model,
  'dbh' => $dbh
};

# model sources
$tt->process('Model.hpp.tt', $vars, "$outdir/${name}Model.hpp")
        || die $tt->error(), "\n";

# node sources
$sth->execute;
while ($vars = $sth->fetchrow_hashref) {
  $name = ucfirst($$vars{'Name'});
  $tt->process('Node.hpp.tt', $vars, "$outdir/${name}Node.hpp")
      || die $tt->error(), "\n";
  $tt->process('Node.cpp.tt', $vars, "$outdir/${name}Node.cpp")
      || die $tt->error(), "\n";
}
$sth->finish;

# wrap up
#$dbh->disconnect;
undef $dbh;
# ^ SQLite driver gives warnings about active statement handles unless closed
#   this way

##
## Filter expression to standard CUDA expression.
##
## @param exp Expression.
##
## @return Filtered expression.
##
sub cudaexp {
  my $exp = shift;
  my $func;

  # transform functions to CUDA_*(...) macros
  $exp =~ s/(\w+)(\s*\()/&cudafunc($1) . $2/ge;

  # translate variables with dimension references
  $exp =~ s/(\w+\[.+?\])/&cudavar($1)/ge;

  # wrap literal floats with REAL(...)
  $exp =~ s/(\b\d*\.\d+(?:[eE][+\-]\d+)?[fl]?)\b/REAL($1)/g;

  return $exp;
}

##
## Map function to CUDA function.
##
sub cudafunc {
  my $key = shift;
  my %funcs = (
      'abs' => 'CUDA_ABS',
      'log' => 'CUDA_LOG',
      'exp' => 'CUDA_EXP',
      'max' => 'CUDA_MAX',
      'min' => 'CUDA_MIN',
      'sqrt' => 'CUDA_SQRT',
      'pow' => 'CUDA_POW',
      'fmod' => 'CUDA_FMOD',
      'modf' => 'CUDA_MODF',
      'ceil' => 'CUDA_CEIL',
      'floor' => 'CUDA_FLOOR'
  );

  if (exists $funcs{$key}) {
      return $funcs{$key};
  } else {
      warn("No CUDA function translation found for $key");
      return $key;
  }
}

##
## Map variable with square-backet dimension reference to alphanumeric
## variable name.
##
sub cudavar {
  my $var = shift;
  my $name;
  my $ord;
  my @ords;
  my $dim;
  my $offset;

  $var =~ /(\w+)\s*\[\s*(.+)\s*\]/;
  $name = $1;
  $ord = $2;
  @ords = split(/\s*,\s*/, $ord);
  foreach $ord (@ords) {
      $ord =~ /([xyz])\s*([+\-])?\s*(\d+)?/
          || warn("Reference in formula not understood: $var");
      if (int($3) != 0) {
	  $name .= "_$1";
	  $name .= ($2 eq '+') ? 'p' : 'm';
	  $name .= $3;
      }
  }

  return $name;
}

##
## Filter expression to LaTeX math expression.
##
## @param exp Expression.
##
## @return Filtered expression.
##
## @todo Need to handle floor, ceil, pow and subscript/superscript, will need
## more than regular expressions to handle nested parentheses etc.
##
sub latexexp {
  my $exp = shift;
  my $key;

  # function mappings
  my %funcs = {
      'abs' => '\abs',
      'log' => '\log',
      'exp' => '\exp',
      'max' => '\max',
      'min' => '\min',
      'sqrt' => '\sqrt',
      'pow' => '\pow',
      'fmod' => '\mod',
      'modf' => '\mod'
  };

  # function to delimiter mappings

  my %funcl = {
      'ceil' => '\lceil',
      'floor' => '\lfloor'
  };
  my %funcr = {
      'ceil' => '\rceil',
      'floor' => '\rfloor'
  };

  # operator mappings
  my %ops = {
      '*' => ' '
  };

  # transform functions, ops, etc
  foreach $key (keys %funcs) {
      $exp =~ s/\b$key(\s*\()\b/$funcs{$key}$1/g;
  }
  foreach $key (keys %ops) {
      $exp =~ s/\b$key\b/$ops{$key}/g;
  }

  return $exp;
}


__END__

=head1 NAME

sql2cpp -- C++ code generator from SQLite model specification.

=head1 SYNOPSIS

sql2cpp --model model.db --ttdir ... --outdir ...

The SQLite database is read from the given *.db file in model, templates from
ttdir and C++ source files written to outdir.

=head1

=over 10

=item B<--help>

Print a brief help message and exit.

=item B<--model>

Specify the database file name.

=item B<--outdir>

Specify the output directory for C++ source files.

=item B<--ttdir>

Specify directory containing templates.

=back

=head1 DESCRIPTION

Reads an SQLite database model specification and generates C++ code
implementing the model for the bi library.

=cut
