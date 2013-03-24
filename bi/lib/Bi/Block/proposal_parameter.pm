=head1 NAME

parameter - declare the proposal density over parameters.

=head1 SYNOPSIS

    sub proposal_parameter {
      ...
    }
    
=head1 DESCRIPTION

Use the C<proposal_parameter> block to specify the proposal density over the
parameters of the model.

Actions in the C<proposal_parameter> block may only refer to variables of
type C<input> and C<param>. They may only target variables of type C<param>.

=cut

package Bi::Block::proposal_parameter;

use base 'Bi::Model::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
