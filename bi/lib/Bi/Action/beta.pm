=head1 NAME

beta - beta distribution.

=head1 SYNOPSIS

    x ~ beta()
    x ~ beta(1.0, 1.0)
    x ~ beta(alpha = 1.0, beta = 1.0)

=head1 DESCRIPTION

A C<beta> action specifies a variate that is beta distributed according to
the given C<alpha> and C<beta> parameters.

=cut

package Bi::Action::beta;

use base 'Bi::Model::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item * C<alpha> (position 0, default 1.0)

First shape parameter of the distribution.

=item * C<beta> (position 1, default 1.0)

Second shape parameter of the distribution.

=back

=cut
our $ACTION_ARGS = [
  {
    name => 'alpha',
    positional => 1,
    default => 1.0
  },
  {
    name => 'beta',
    positional => 1,
    default => 1.0
  }
];

sub validate {
    my $self = shift;
    
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('~');
    $self->ensure_scalar('alpha');
    $self->ensure_scalar('beta');
    $self->set_parent('beta_');
    $self->set_can_combine(1);
    $self->set_unroll_args(0);
}

sub mean {
    my $self = shift;

    # shape*scale    
    my $alpha = $self->get_named_arg('alpha')->clone;
    my $beta = $self->get_named_arg('beta')->clone;
    my $mean = $alpha/($alpha + $beta);

    return $mean;
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev: 2921 $ $Date: 2012-08-12 13:49:45 +0800 (Sun, 12 Aug 2012) $
