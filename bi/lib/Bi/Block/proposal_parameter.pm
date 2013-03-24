=head1 NAME

parameter - a proposal distribution over parameters.

=head1 SYNOPSIS

    sub proposal_parameter {
      theta ~ gaussian(theta, 1.0)  // local proposal
      theta ~ gaussian(0.0, 1.0)    // independent proposal
    }
    
=head1 DESCRIPTION

This may be a local or independent proposal distribution, used by the
C<sample> command.

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
