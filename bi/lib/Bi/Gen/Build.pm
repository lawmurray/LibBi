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
use File::Spec;

use Bi qw(share_file share_dir);
use Bi::Gen;
use Bi::Model;
use Bi::Expression;

=item B<new>(I<outdir>)

Constructor.

=cut
sub new {
    my $class = shift;
    my $outdir = shift;

    my $ttdir = share_dir(File::Spec->catdir('tt', 'build'));
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
    assert(!defined $model || $model->isa('Bi::Model')) if DEBUG;

    $self->process_template('Makefile.am.tt', {
        'have_model' => defined $model,
        'model' => $model
    }, 'Makefile.am');
    $self->copy_file('autogen.sh', 'autogen.sh');
    $self->copy_file('configure.ac', 'configure.ac');
    $self->copy_file('nvcc_wrapper.pl', 'nvcc_wrapper.pl');
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
