=head1 NAME

uninformative - uninformative distribution.

=head1 SYNOPSIS

    x ~ uninformative()

=head1 DESCRIPTION

An C<uninformative> action indicates that a variable has an uninformative
distribution.

An C<uninformative> action may only be used within a L<parameter> block. The
presence of any one such action will preclude sampling from the prior
distribution, although densities will still be computable.

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
    $self->set_parent('uninformative_');
    $self->set_can_combine(1);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
