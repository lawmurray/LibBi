=head1 NAME

parameter - declare the prior density over parameters.

=head1 SYNOPSIS

    sub parameter {
      ...
    }
    
=head1 DESCRIPTION

Use the C<parameter> block to specify the prior density over the parameters
of the model.

Actions in the C<parameter> block may only target and refer to variables of
type C<param>.

=cut

package Bi::Block::parameter;

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
