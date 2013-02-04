=head1 NAME

rewrite - output new model specification after applying transformations and
optimisations.

=head1 SYNOPSIS

    bi rewrite --model I<model.bi>

=head1 DESCRIPTION

The C<rewrite> command takes a model specification and outputs a new
(pseudo-)specification that shows the internal transformations and
optimisations applied by Bi. It is useful for validation and debugging
purposes. The (pseudo-)specification is written to C<stdout>.

Note that the output is not a correct model specification that can be
re-input to Bi. In particular, element-wise operations (e.g. C<.*>, C<./>)
are converted to their scalar equivalents (e.g. C<*>, C</>) after matrix
and vector operations are unrolled into actions (e.g. C<*> may unroll to the
L<gemv> action). They are therefore unrecoverable in producing the output.

=head1 OPTIONS

The following options are supported:

=over 4

=item C<--model-file>

the model specification file.

=back

=cut

package Bi::Client::rewrite;

use base 'Bi::Client';
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
