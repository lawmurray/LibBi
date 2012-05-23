=head1 NAME

parameter_post_ - declare post-computes for the parameter block.

=head1 SYNOPSIS

    sub parameter_post_ {
      ...
    }
    
=head1 DESCRIPTION

Use the C<parameter_post_> block to perform precomputations of static
expressions immediately after evaluation of the L<parameter> block.

Actions in the C<parameter_post_> block may only target variables of type
C<param_aux_> and refer to variables of type C<param> and C<param_aux_>.

=cut

package Bi::Block::parameter_post_;

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
