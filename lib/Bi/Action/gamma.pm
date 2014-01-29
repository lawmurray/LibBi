=head1 NAME

gamma - gamma distribution.

=head1 SYNOPSIS

    x ~ gamma()
    x ~ gamma(2.0, 5.0)
    x ~ gamma(shape = 2.0, scale = 5.0)

=head1 DESCRIPTION

A C<gamma> action specifies that a variable is gamma distributed according to
the given C<shape> and C<scale> parameters.

=cut

package Bi::Action::gamma;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<shape> (position 0, default 1.0)

Shape parameter of the distribution.

=item C<scale> (position 1, default 1.0)

Scale parameter of the distribution.

=back

=cut
our $ACTION_ARGS = [
  {
    name => 'shape',
    positional => 1,
    default => 1.0
  },
  {
    name => 'scale',
    positional => 1,
    default => 1.0
  }
];

sub validate {
    my $self = shift;
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('~');
    $self->ensure_scalar('shape');
    $self->ensure_scalar('scale');

    unless ($self->get_left->get_shape->equals($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('pdf_');
    $self->set_can_combine(1);
    $self->set_unroll_args(0);
}

sub mean {
    my $self = shift;

    # shape*scale    
    my $shape = $self->get_named_arg('shape')->clone;
    my $scale = $self->get_named_arg('scale')->clone;
    my $mean = $shape*$scale;

    return $mean;
}

sub jacobian {
    my $self = shift;
    
    my $mean = $self->mean;
    my @refs = @{$mean->get_all_var_refs};
    my @J = map { $mean->d($_) } @refs;

    return (\@J, \@refs);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
