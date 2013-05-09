=head1 NAME

draw - draw a model as a directed graph.

=head1 SYNOPSIS

    bi draw --model-file I<Model>.bi > I<Model>.dot
    dot -Tpdf -o I<Model>.pdf I<Model>.dot

=head1 DESCRIPTION

The C<draw> command takes a model specification and outputs a directed graph
to visualise the model. It is useful for validation and debugging
purposes. The output is a C<dot> script that can be processed by the C<dot>
program to create a figure.

=cut

package Bi::Client::draw;

use parent 'Bi::Client';
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
