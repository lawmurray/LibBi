#!/usr/bin/env perl

##
## Bootstraps environment for model.
##
## @author Lawrence Murray <lawrence.murray@csiro.au>
## $Rev$
## $Date$
##

use strict;
use Getopt::Long;
use FindBin qw($Bin);

my $SRCDIR = 'src';
my $MODELDIR = "$SRCDIR/model";
my $BUILDDIR = 'build';
my $RESULTSDIR = 'results';
my $OLDDIR = 'old';

# command line arguments
my $model = '';
Getopt::Long::Configure("auto_help");
GetOptions("model=s" => \$model);

$model =~ /(\w+)\.csv/i;
my $name = ucfirst($1) || die('Model name empty');

print "Saving previous config...\n";
system("make save");

print "Cleaning...\n";
system("make clean");

print "Setting up directories...\n";
system("mkdir -p $SRCDIR $BUILDDIR $RESULTSDIR");

print "Interpreting model...\n";
system("$Bin/csv2sql.pl --model $name --outdir $BUILDDIR < $model");

print "Visualising model...";
if (system('dot < /dev/null') == 0) {
    system("$Bin/sql2dot.pl --model $name --dbfile $BUILDDIR/$name.db > $BUILDDIR/$name.dot");
    system("dot -Tpdf -o $name.pdf < $BUILDDIR/$name.dot");
} else {
    print " dot not found, skipping";
}
print "\n";

print "Generating model code...\n";
system("$Bin/sql2cpp.pl --outdir $MODELDIR --model $BUILDDIR/$name.db");

print "Generating client code...\n";
system("$Bin/sql2client.pl --outdir $SRCDIR --model $BUILDDIR/$name.db");

print "Generating Makefile...\n";
system("perl $Bin/sql2make.pl $name > Makefile");

print "Copying compile config from library...\n";
system("cp $Bin/../../lib/config.mk .");

print "Generating scripts...\n";
system("$Bin/sql2sh.pl --outdir . --model $BUILDDIR/$name.db");
system("chmod +x *.sh");

print "Restoring previous config...\n";
system("make restore");
