=head1 NAME

uniform - uniform distribution.

=head1 SYNOPSIS

    x ~ uniform()
    x ~ uniform(0.0, 1.0)
    x ~ uniform(lower = 0.0, upper = 1.0)

=head1 DESCRIPTION

A C<uniform> action specifies that a variable is uniformly distributed on a
finite and left-closed interval given by the bounds C<lower> and C<upper>.

=cut

package Bi::Action::uniform;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<lower> (position 0, default 0.0)

Lower bound on the interval.

=item C<upper> (position 1, default 1.0)

Upper bound on the interval.

=back

=cut
our $ACTION_ARGS = [
  {
    name => 'lower',
    positional => 1,
    default => 0.0
  },
  {
    name => 'upper',
    positional => 1,
    default => 1.0
  }
];

=head1 METHODS

=item B<make_range>

Construct an expression giving the range of the distribution.

=cut
sub make_range {
    my $self = shift;
    my $lower = $self->get_named_arg('lower')->clone;
    my $upper = $self->get_named_arg('upper')->clone;
    my $range = $upper - $lower;
    
    return $range->simplify;
}

sub validate {
    my $self = shift;
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    
    $self->ensure_op('~');
    $self->ensure_scalar('lower');
    $self->ensure_scalar('upper');

    unless ($self->get_left->get_shape->compat($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('pdf_');
    $self->set_can_combine(1);
    $self->set_unroll_args(0);
}

sub mean {
    my $self = shift;
    
    # (lower + upper)/2
    my $lower = $self->get_named_arg('lower')->clone;
    my $upper = $self->get_named_arg('upper')->clone;
    my $mean = 0.5*($lower + $upper);

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

=head1 VERSION

$Rev$ $Date$
