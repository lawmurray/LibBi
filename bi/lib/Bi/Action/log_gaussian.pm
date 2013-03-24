=head1 NAME

log_gaussian - log-Gaussian distribution.

=head1 SYNOPSIS

    x ~ log_gaussian()
    x ~ log_gaussian(0.0, 1.0)
    x ~ log_gaussian(mean = 0.0, std = 1.0)

=head1 DESCRIPTION

A C<log_gaussian> action specifies that the logarithm of a variable is
Gaussian distributed according to the given C<mean> and C<std> parameters.

=cut

package Bi::Action::log_gaussian;

use base 'Bi::Action::gaussian';

=head1 PARAMETERS

=over 4

=item C<mean> (position 0, default 0.0)

Mean of the log-transformed variable.

=item C<std> (position 1, default 1.0)

Standard deviation of the log-transformed variable.

=back

=cut

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
