=head1 NAME

observation - declare the observation density.

=head1 SYNOPSIS

    sub observation {
      ...
    }
    
=head1 DESCRIPTION

Use the C<observation> block to specify the observation density of the model.

Actions in the C<observation> block may only refer to variables of type
C<param>, C<force> and C<state>. They may only target variables of type
C<obs>, using the C<~> operator.

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
