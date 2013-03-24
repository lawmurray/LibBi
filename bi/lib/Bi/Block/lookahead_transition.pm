=head1 NAME

lookahead_transition - declare a lookahead density to accompany the
transition density.

=head1 SYNOPSIS

    sub lookahead_transition {
      ...
    }
    
=head1 DESCRIPTION

Use the C<lookahead_transition> block to specify a lookahead density to
accompany the transition density. This may be a deterministic or
computationally cheaper version of the transition density, for example. It is
used by methods such as the auxiliary particle filter. 

Actions in the C<lookahead_transition> block may only target variables of
type:

=over 4

=item * C<noise>, using the C<~> operator, referring only to variables of
type C<param> and C<input>, and

=item * C<state>, using the C<<-> operator, referring only to variables of
type C<param>, C<input>, C<state> and C<noise>.

=back

=cut

package Bi::Block::lookahead_transition;

use base 'Bi::Model::Block';
use warnings;
use strict;

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
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
