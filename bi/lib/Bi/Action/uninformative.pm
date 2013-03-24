=head1 NAME

uninformative - uninformative distribution.

=head1 SYNOPSIS

    x ~ uninformative()

=head1 DESCRIPTION

An C<uninformative> action specifies that a variable has an uninformative
distribution.

The use of an C<uninformative> action in a block precludes sampling from that
block, although densities may still be computed. If used in the L<parameter>
or L<initial> block, a L<proposal_parameter> or L<proposal_initial> block
should be used for the C<sample> command to work.

=cut

package Bi::Action::uninformative;

use base 'Bi::Model::Action';
use warnings;
use strict;

our $ACTION_ARGS = [];

sub validate {
    my $self = shift;
    
    $self->process_args($ACTION_ARGS);
    
    $self->ensure_op('~');
    $self->set_parent('pdf_');
    $self->set_can_combine(1);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
