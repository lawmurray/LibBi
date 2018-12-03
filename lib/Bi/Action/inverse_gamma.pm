=head1 NAME

inverse_gamma - inverse gamma distribution.

=head1 SYNOPSIS

    x ~ inverse_gamma()
    x ~ inverse_gamma(2.0, 1.0/5.0)
    x ~ inverse_gamma(shape = 2.0, scale = 1.0/5.0)

=head1 DESCRIPTION

An C<inverse_gamma> action specifies that a variable is inverse-gamma
distributed according to the given C<shape> and C<scale> parameters.

=cut

package Bi::Action::inverse_gamma;

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

    unless ($self->get_left->get_shape->compat($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('pdf_');
    $self->set_can_combine(1);
    $self->set_unroll_args(0);
}

sub mean {
    my $self = shift;
    
    # shape*(scale^2)
    my $shape = $self->get_named_arg('shape')->clone;
    my $scale = $self->get_named_arg('scale')->clone;
    my $mean = $shape*(new Bi::Expression::Function('pow', [ $scale, new Bi::Expression::Literal(2.0) ]));    
    
    return $mean->simplify;
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

