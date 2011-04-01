#!/usr/bin/env perl

##
## Create client program code for specific SQLite model specification.
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
my $ttdir = "$Bin/../tt/sh";
Getopt::Long::Configure("auto_help");
GetOptions("model=s" => \$model,
    "outdir=s" => \$outdir,
    "ttdir=s" => \$ttdir);

$model =~ /(\w+)\.db/i;
my $name = ucfirst($1);

# connect to database
$ENV{'DBI_DSN'} = "dbi:SQLite:dbname=$model";
my $dbh = DBI->connect($ENV{'DBI_DSN'});
my $sth = $dbh->prepare("SELECT Name FROM Node " .
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
  'LibPath' => "$Bin/../../lib/build",
  'dbh' => $dbh
};

# client program sources
my $prog;
foreach $prog ('simulate', 'pf', 'ukf', 'urts', 'mcmc', 'likelihood',
	       'config', 'predict') {
    $tt->process("$prog.sh.tt", $vars, "$outdir/$prog.sh")
	|| die $tt->error(), "\n";
}

# wrap up
#$dbh->disconnect;
undef $dbh;
# ^ SQLite driver gives warnings about active statement handles unless closed
#   this way

__END__

=head1 NAME

sql2client -- C++ code generator from SQLite model specification for bi client
programs.

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
