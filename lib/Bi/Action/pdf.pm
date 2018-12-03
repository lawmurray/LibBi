=head1 NAME

pdf - arbitrary probability density function.

=head1 SYNOPSIS

    x ~ I<expression>
    x ~ pdf(pdf = I<expression>, max_pdf = I<expression>)
    x ~ pdf(pdf = I<log_expression>, log = 1)

=head1 DESCRIPTION

A C<pdf> action specifies that a variable is distributed according to some
arbitrary probability density function. It need not be used explicitly
unless a maximum probability density function needs to be supplied with it:
any expression using the C<~> operator without naming an action is evaluated
using C<pdf>.

=cut

package Bi::Action::pdf;

use parent 'Bi::Action';
use warnings;
use strict;

=head1 PARAMETERS

=over 4

=item C<pdf> (position 0)

An expression giving the probability density function.

=item C<max_pdf> (position 1, default inf)

An expression giving the maximum of the probability density function.

=item C<log> (default 0)

Is the expression given the log probability density function?

=back

=cut
our $ACTION_ARGS = [
  {
    name => 'pdf',
    positional => 1
  },
  {
    name => 'max_pdf',
    default => 0.0,
    positional => 1
  },
  {
  	name => 'log',
  	default => 0
  }
];

sub validate {
    my $self = shift;
    
    Bi::Action::validate($self);
    $self->process_args($ACTION_ARGS);
    $self->ensure_op('~');
    $self->ensure_scalar('pdf');
    $self->ensure_scalar('max_pdf');
    $self->ensure_scalar('log');
    $self->ensure_const('log');

    unless ($self->get_left->get_shape->compat($self->get_shape)) {
    	die("incompatible sizes on left and right sides of action.\n");
    }

    $self->set_parent('pdf_');
    $self->set_can_combine(1);
    $self->set_unroll_args(0);
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

