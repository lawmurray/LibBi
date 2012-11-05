=head1 NAME

matrix_ - matrix expression block.

=head1 DESCRIPTION

This block behaves the same as L<eval_>, but is required to group matrix
actions into separate blocks from scalar actions.

=cut

package Bi::Block::matrix_;

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

$Rev: 2867 $ $Date: 2012-07-31 15:38:06 +0800 (Tue, 31 Jul 2012) $
