=head1 NAME

observation - the likelihood function.

=head1 SYNOPSIS

    sub observation {
      ...
    }
    
=head1 DESCRIPTION

Actions in the C<observation> block may only refer to variables of type
C<param>, C<input> and C<state>. They may only target variables of type
C<obs>.

=cut

package Bi::Block::observation;

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
