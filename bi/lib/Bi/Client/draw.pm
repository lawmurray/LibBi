=head1 NAME

draw - visualise a model specification as a directed graph.

=head1 SYNOPSIS

    bi draw --model-file I<model.bi> > I<model.dot>
    dot -Tpdf -o I<model.pdf> I<model.dot>

=head1 DESCRIPTION

The C<draw> command takes a model specification and outputs a directed graph
that visualises the model. It is useful for validation and debugging
purposes. The output is in the format of a DOT script. It will need to be
processed by the C<dot> program in order to create an image (see example
above).

=cut

package Bi::Client::draw;

use base 'Bi::Client';
use warnings;
use strict;

use Bi::Gen::Dot;

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

sub needs_transform {
    return 0;
}

sub exec {
    my $self = shift;
    my $model = shift;
    
    my $dot = new Bi::Gen::Dot();
    $dot->gen($model);
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
