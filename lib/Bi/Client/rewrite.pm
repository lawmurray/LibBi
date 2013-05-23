=head1 NAME

rewrite - output internal model representation after applying transformations
and optimisations.

=head1 SYNOPSIS

    libbi rewrite --model-file I<Model>.bi

=head1 DESCRIPTION

The C<rewrite> command takes a model specification and outputs a new
specification that shows the internal transformations and
optimisations applied by LibBi. It is useful for validation and debugging
purposes. The new specification is written to C<stdout>.

=cut

package Bi::Client::rewrite;

use parent 'Bi::Client';
use warnings;
use strict;

use Bi::Gen::Bi;

=head1 METHODS

=over 4

=cut

sub init {
    my $self = shift;

    $self->{_binary} = undef;
}

sub is_cpp {
    return 0;
}

sub exec {
    my $self = shift;
    my $model = shift;
    
    my $bi = new Bi::Gen::Bi();
    $bi->gen($model);
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
