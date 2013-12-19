=head1 NAME

test_filter - test filter and print diagnostics.

=head1 SYNOPSIS

    libbi test_filter ...
    
=head1 INHERITS

L<Bi::Client::filter>

=cut

package Bi::Test::test_filter;

use parent 'Bi::Client::filter';
use warnings;
use strict;

=head1 OPTIONS

The C<test_filter> command inherits all options from C<filter>, and permits
the following additional options:

=over 4

=item C<--nruns> (default 1)

Number of samples to draw.

=item C<--nobs> (default 1)

Number of observation sets, starting from C<--obs-np> in C<--obs-file>, on
which to test.

=back

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'nruns',
      type => 'int',
      default => 1
    },
    {
      name => 'nobs',
      type => 'int',
      default => 1
    }
);

sub init {
    my $self = shift;

    Bi::Client::filter::init($self);
    push(@{$self->{_params}}, @CLIENT_OPTIONS);
}

sub process_args {
    my $self = shift;

    $self->Bi::Client::process_args(@_);
    my $filter = $self->get_named_arg('filter');
    my $binary;
    if ($filter eq 'kalman') {
        $self->set_named_arg('with-transform-extended', 1);
    } else {
        $binary = 'test_filter';
    }
    $self->{_binary} = $binary;
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>
