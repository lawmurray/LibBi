=head1 NAME

optimise - optimisation of the parameters of a model.

=head1 SYNOPSIS

    libbi optimise ...
    
    libbi optimize ...

Alternative spellings are supported.

=head1 INHERITS

L<Bi::Client::filter>

=cut

package Bi::Client::optimise;

use parent 'Bi::Client::filter';
use warnings;
use strict;

=head1 OPTIONS

The C<optimise> command inherits all options from L<filter>, and permits the
following additional options:

=over 4

=item C<--target> (default C<likelihood>)

Optimisation target; one of:

=over 8

=item C<likelihood>

Maximum likelihood estimation.

=item C<posterior>

Maximum I<a posteriori> estimation.

=back

=item C<--optimiser> or C<--optimizer> (default C<nm>)

The optimisation method to use; one of:

=over 8

=item C<nm>

Nelder-Mead simplex method.

=back

=back

=head2 Nelder-mead simplex method-specific options

=over 4

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
      deprecated => 1,
      message => 'use --target instead'  
    },
    {
      name => 'target',
      type => 'string',
      default => 'likelihood'
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
    $self->{_binary} = 'optimise';
}

1;

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

