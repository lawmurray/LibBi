=head1 NAME

transition - the transition distribution.

=head1 SYNOPSIS

    sub transition(delta = 1.0) {
      ...
    }
    
=head1 DESCRIPTION

Actions in the C<transition> block may reference variables of any
type, but may only target variables of type C<noise> and C<state>.

=cut

package Bi::Block::transition;

use base 'Bi::Model::Block';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<delta> (position 0, default 1.0)

The time step for discrete-time components of the transition. Must be a
constant expression.

=back

=cut

our $BLOCK_ARGS = [
  {
    name => 'delta',
    positional => 1,
    default => 1.0
  }
];

sub validate {
    my $self = shift;
    
    $self->process_args($BLOCK_ARGS);
    if (!$self->get_named_arg('delta')->is_const) {
        die("argument 'delta' to block 'transition' must be a constant expression\n");
    }
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
