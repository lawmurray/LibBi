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

=item C<--Ps> (default 1)

Number of sizes to use. The first size is set to the value of C<--nparticles>,
each subsequent size multiples this by two.

=item C<--reps> (default 100)

Number of trials on each size.



=back

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'Ps',
      type => 'int',
      default => 1
    },
    {
      name => 'reps',
      type => 'int',
      default => 100
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
    }
    $binary = 'test_filter';
    $self->{_binary} = $binary;
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>
