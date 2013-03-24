=head1 NAME

initial - the prior distribution over the initial values of state variables.

=head1 SYNOPSIS

    sub initial {
      ...
    }
    
=head1 DESCRIPTION

Actions in the C<initial> block may only refer to variables of type
C<param>, C<input> and C<state>. They may only target variables of type
C<state>.

=cut

package Bi::Block::initial;

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
