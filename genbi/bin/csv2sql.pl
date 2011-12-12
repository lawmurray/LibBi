#!/usr/bin/env perl

##
## Reads a CSV model specification and populates an SQLite database.
##
## @author Lawrence Murray <lawrence.murray@csiro.au>
## $Rev$
## $Date$
##

use strict;
use Text::CSV;
use DBI;
use DBD::SQLite;
use Pod::Usage;
use Getopt::Long;
use List::MoreUtils qw(uniq);
use FindBin qw($Bin);

# command line arguments
my $model = '';
my $outdir = '.';
my $srcdir = "$Bin/../src";
Getopt::Long::Configure("auto_help");
GetOptions("model=s" => \$model,
    "outdir=s" => \$outdir,
    "srcdir=s" => \$srcdir);

# create sqlite database
my $dbfile = "$outdir/$model.db";
`sqlite3 $dbfile < $srcdir/sqlite.sql`;

# connect to database
my $dbh = DBI->connect("dbi:SQLite:dbname=$dbfile", '', '', {
    AutoCommit => 0,
    PrintError => 0
});

# database statement handles
my %sth;
$sth{'InsertNode'} = $dbh->prepare('INSERT INTO Node(Name,' .
    'Description,Category,HasX,HasY,HasZ,Position) VALUES (?,?,?,?,?,?,?)');
$sth{'InsertNodeTrait'} = $dbh->prepare('INSERT INTO NodeTrait(Node,' .
    'Trait) VALUES (?,?)');
$sth{'InsertNodeFormula'} = $dbh->prepare('INSERT INTO NodeFormula(Node,' .
    'Function,Formula,XOrdinate,YOrdinate,ZOrdinate,Position) ' .
    'VALUES (?,?,?,?,?,?,?)');
$sth{'InsertEdge'} = $dbh->prepare('INSERT INTO Edge(ParentNode,ChildNode,' .
    'XOffset,YOffset,ZOffset,Position) VALUES (?,?,?,?,?,?)');
$sth{'UpdatePosition'} = $dbh->prepare('UPDATE Node SET Position = ? WHERE ' .
    'Name = ?');
$sth{'GetNodes'} = $dbh->prepare("SELECT Name FROM Node WHERE Category IN (" .
    "'Observation', 'Static variable', 'Discrete-time variable', " .
    "'Continuous-time variable')");
$sth{'InsertParent'} = $dbh->prepare('INSERT INTO Parent(ParentNode,' .
    'ChildNode,XOffset,YOffset,ZOffset,Position) VALUES (?,?,?,?,?,?)');

# process CSV headers
my $io = \*STDIN;
my $csv = Text::CSV->new();
$csv->column_names($csv->getline($io));

# process CSV
my $fields;
my $val;
my $pos1;
my $pos2;
my @nodes;
my @parents;
my @children;
my @positions;
my %dependents;
my $i;
my $j;
my %formulae;
my $function;
my $formula;
my %dimensions;
my %ords;
my @ords;
my $ords;
my $ord;
my $dim;
my $parent;

$fields = $csv->getline_hr($io);
$pos1 = 0;
while (!$csv->eof()) {
  push(@nodes, $$fields{'Name'});

  undef %dimensions;
  map { $dimensions{$_} = 1 } split /\s*;\s*/, $$fields{'Dimensions'};

  # insert node
  $sth{'InsertNode'}->execute($$fields{'Name'},
      $$fields{'Description'}, $$fields{'Category'},
      int(exists $dimensions{'x'}), int(exists $dimensions{'y'}),
      int(exists $dimensions{'z'}), $pos1) ||
      warn("Problem with node $$fields{'Name'}");

  # insert traits
  foreach $val (split /;\s*/, $$fields{'Traits'}) {
    $sth{'InsertNodeTrait'}->execute($$fields{'Name'}, uc($val)) ||
        warn("Problem with trait $val of node $$fields{'Name'}");
  }

  # insert functions and formulae
  undef %formulae;
  undef %ords;
  $pos2 = 0;
  foreach $val (split /\s*;\s*/, $$fields{'Formulae'}) {
    if ($val =~ /\s*(\w+)\s*(?:\[\s*([A-Za-z0-9, ]+)\s*\])?\s*:\s*(.*)/) {
      $function = $1;
      $ords = $2;
      $formula = $3;

      @ords = split(/\s*,\s*/, $ords);

      $ords[0] = -1 if ($ords[0] eq 'x' || $ords[0] eq '');
      $ords[1] = -1 if ($ords[1] eq 'y' || $ords[1] eq '');
      $ords[2] = -1 if ($ords[2] eq 'z' || $ords[2] eq '');

      $sth{'InsertNodeFormula'}->execute($$fields{'Name'}, $function,
        $formula, int($ords[0]), int($ords[1]), int($ords[2]), $pos2);
      ++$pos2;
    } else {
      warn("Syntax error in formulae of node $$fields{'Name'}");
    }
  }

  # store edges for later
  $pos2 = 0;
  %{$dependents{$$fields{'Name'}}} = ();
  foreach $val (split /\s*;\s*/, $$fields{'Dependencies'}) {
    push(@parents, $val);
    push(@children, $$fields{'Name'});
    push(@positions, $pos2);

    $val =~ s/\s*\[.*?\]//g;
    if ($$fields{'Traits'} =~ /\bIS_S_NODE\b/) {
      # need to topologically order
      $dependents{$$fields{'Name'}}{$val} = 1;
    }
    ++$pos2;
  }

  # next
  ++$pos1;
  $fields = $csv->getline_hr($io);
}

# insert dependencies as edges
for ($i = 0; $i < @parents; ++$i) {
  $parents[$i] =~ /(\w+)\s*(?:\[\s*([A-Za-z0-9 \-+,]+)\s*\])?/;
  $parent = $1;
  @ords = split(/\s*;\s*/, $2);

  $ords{'x'} = 0;
  $ords{'y'} = 0;
  $ords{'z'} = 0;
  foreach $ord (@ords) {
    $ord =~ /(\w+)\s*([-+]\s*\d+)?/;
    $dim = $1;
    $ord = $2;
    $ord =~ s/ //g;
    $ord = int($ord);
    $ords{$dim} = $ord;
  }
  
  $sth{'InsertEdge'}->execute($parent, $children[$i], $ords{'x'}, $ords{'y'},
      $ords{'z'}, $positions[$i]) ||
      warn("Problem with dependency $parents[$i] of node $children[$i]\n");
}

# commit first phase
$dbh->commit;

# inline intermediate results to flatten parent list
$sth{'GetNodes'}->execute;
while ($fields = $sth{'GetNodes'}->fetchrow_hashref) {
  &Parents($$fields{'Name'});
}

# wrap up 
$dbh->commit;

my $key;
foreach $key (keys %sth) {
  $sth{$key}->finish;
}
undef %sth; # workaround for SQLite warning about active statement handles
$dbh->disconnect;

##
## Topologically sort s-nodes so that no node appears before all of its parents
## have appeared.
##
## @param dependencies Hash-of-hashes, outside keyed by s-node, inside
## list of parents of that node (of any types). Destroyed in process.
## 
## @return Sorted array of node names.
##
sub TopologicalSort {
  my $nodes = shift;
  my $dependencies = shift;
  my @result;
  my $key;
  my $node;
  my $i;

  # flush self-dependencies and non s-node dependencies
  foreach $key (keys %$dependencies) {
    delete $dependencies->{$key}{$key};
    foreach $node (keys %{$dependencies->{$key}}) {
      if (!exists $dependencies->{$node}) {
        delete $dependencies->{$key}{$node};
      }
    }
  }
  
  # sort
  while (@$nodes) {
    # find node with all dependencies satisfied
    $i = 0;
    while ($i < @$nodes && keys %{$dependencies->{$$nodes[$i]}} > 0) {
      ++$i;
    }
    if ($i >= @$nodes) {
      $i = 0;
      warn('S-nodes have no partial order, loop exists?');
    }

    $node = $$nodes[$i];
    splice(@$nodes, $i, 1);

    push(@result, $node);
    delete $dependencies->{$node};
    
    # delete this node from dependency lists
    foreach $key (keys %$dependencies) {
      delete $dependencies->{$key}{$node};
    }
  }
  
  return @result;
}

##
## Return parents for node, with inlining.
##
## @param name Name of node.
##
## @returns List of edge ids tracing up to all parents via inlined
## intermediate results, without duplication
##
sub Parents {
  my $name = shift;
  my $child;
  my $pos;
  my %base;
  if (@_) {
    $child = shift;
    $pos = shift;
  } else {
    $child = $name;
    $pos = \%base;
  }

  my $parent;
  my $sth = $dbh->prepare('SELECT Node.Name, Node.Category, ' .
    'Edge.XOffset, Edge.YOffset, Edge.ZOffset FROM Node, Edge ' .
    'WHERE Edge.ChildNode = ? AND ' .
    'Edge.ParentNode = Node.Name ORDER BY Edge.Position');

  $sth->execute($name);
  while ($parent = $sth->fetchrow_hashref) {
    if ($$parent{'Category'} eq 'Intermediate result') {
      &Parents($$parent{'Name'}, $child, $pos); # inline
    }
    if ($sth{'InsertParent'}->execute($$parent{'Name'}, $child,
        $$parent{'XOffset'}, $$parent{'YOffset'}, $$parent{'ZOffset'},
				      int($$pos{$$parent{'Category'}}))) {
	++$$pos{$$parent{'Category'}};
    }
  }
  $sth->finish;
}

__END__

=head1 NAME

csv2sql -- CSV to SQLite database converter for model specifications.

=head1 SYNOPSIS

csv2sql [options]

The CSV model specification is read from stdin. The SQLite database will be
written to $model.db, where $model is specified as a command line argument.

=head1 

=over 10

=item B<--help>

Print a brief help message and exit.

=item B<--model>

Specify the model name.

=item B<--outdir>

Specify the output directory.

=item B<--srcdir>

Specify the source directory (the directory in which the sqlite.sql script
resides).

=back

=head1 DESCRIPTION

Reads a CSV model specification from stdin and generates an SQLite database
representation of the same.

=cut

