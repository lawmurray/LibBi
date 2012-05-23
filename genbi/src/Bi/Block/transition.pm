=head1 NAME

transition - declare the transition density.

=head1 SYNOPSIS

    sub transition(delta = 1.0) {
      ...
    }
    
=head1 DESCRIPTION

Use the C<transition> block to specify the transition density.

Actions in the C<transition> block may only target variables of type:

=over 4

=item * C<noise>, using the C<~> operator, referring only to variables of
type C<param> and C<force>, and

=item * C<state>, using the C<<-> operator, referring only to variables of
type C<param>, C<force>, C<state> and C<noise>.

=back

=head1 PARAMETERS

=over 4

=item * C<delta> (default 1.0)

The time step for discrete-time components of the transition.

=back

=cut

package Bi::Block::transition;

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
