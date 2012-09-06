=head1 NAME

pdf_ - block for univariate pdf actions.

=cut

package Bi::Block::pdf_;

use base 'Bi::Model::Block';
use warnings;
use strict;

our $BLOCK_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    
    if ($self->num_blocks > 0) {
        die("a 'pdf_' block may not contain sub-blocks\n");
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev: 2925 $ $Date: 2012-08-12 17:49:28 +0800 (Sun, 12 Aug 2012) $
