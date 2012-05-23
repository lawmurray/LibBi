=head1 NAME

Bi::Gen::Build - generator for GNU Autotools build system

=head1 SYNOPSIS

    use Bi::Gen::Build;
    $gen = new Bi::Gen::Build($outdir);
    $gen->gen($model);

=head1 INHERITS

L<Bi::Gen>

=head1 METHODS

=over 4

=cut

package Bi::Gen::Build;

use base 'Bi::Gen';
use warnings;
use strict;

use Carp::Assert;
use FindBin qw($Bin);
use File::Spec;

use Bi::Gen;
use Bi::Model;
use Bi::Expression;

=item B<new>(I<outdir>)

Constructor.

=cut
sub new {
    my $class = shift;
    my $outdir = shift;

    my $ttdir = File::Spec->catdir($Bin, '..', 'tt', 'build');
    my $self = Bi::Gen->new($ttdir, $outdir);

    bless $self, $class;    
    return $self;
}

=item B<gen>(I<model>)

Generate code for model.

=cut
sub gen {
    my $self = shift;
    my $model = shift;

    # pre-condition
    assert($model->isa('Bi::Model')) if DEBUG;

    foreach my $file ('autogen.sh', 'configure.ac', 'Makefile.am') {
        $self->process_template("$file.tt", { 'model' => $model }, $file);
    }
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
