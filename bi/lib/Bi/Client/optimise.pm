=head1 NAME

optimise - frontend to optimisation client programs.

=head1 SYNOPSIS

    bi optimise ...
    
Note that C<optimize> may be used as an alternative spelling of C<optimise>.

=head1 INHERITS

L<Bi::Client::filter>

=cut

package Bi::Client::optimise;

use base 'Bi::Client::filter';
use warnings;
use strict;

=head1 OPTIONS

The C<optimise> program inherits all options from L<filter>, and permits the
following additional options:

=over 4

=item C<--optimiser> or C<--optimizer> (default C<'nm'>)

The optimisation method to use; one of:

=over 8

=item C<'nm'>

Nelder-Mead simplex method.

=back

=item C<--mode> (default C<'ml'>)

Optimisation mode, one of:

=over 8

=item C<'ml'>

Maximum likelihood estimation.

=item C<'map'>

Maximum I<a posteriori> estimation.

=back

=item C<--simplex-size-real> (default 0.1)

Size of initial simplex relative to starting point of each variable. 

=item C<--stop-size> (default 1.0-e4)

Size-based stopping criterion.

=item C<--stop-steps> (default 100)

Maximum number of steps to take.

=back

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'optimiser',
      type => 'string',
      default => 'nm'
    },
    {
      name => 'optimizer',
      type => 'string',
    },
    {
      name => 'mode',
      type => 'string',
      default => 'ml'
    },
    {
      name => 'simplex-size-rel',
      type => 'float',
      default => 0.1
    },
    {
      name => 'stop-size',
      type => 'float',
      default => 1.0e-4
    },
    {
      name => 'stop-steps',
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

    $self->Bi::Client::filter::process_args(@_);
    my $binary = $self->get_named_arg('optimiser');
    if ($self->is_named_arg('optimizer') && $self->is_named_arg('optimizer') ne '') {
        # alternate spelling used
        $binary = $self->get_named_arg('optimizer');
    }
    
    $self->{_binary} = $binary;
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
