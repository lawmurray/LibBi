=head1 NAME

Bi::Visitor - abstract class for functors which operate as visitors over
L<Bi::Expression> trees.

=head1 DESCRIPTION

C<Bi::Visitor> objects are functors that operate as visitors over
L<Bi::Expression> trees. These classes have no constructors as such, and are
not intended to be instantiated persistently. To use, simply call the B<evaluate>
method; this will construct a temporary (local) object of the class, visit
the L<Bi::Expression> object passed as an argument, and return the result.

=head1 METHODS

=over 4

=cut

package Bi::Visitor;

use warnings;
use strict;

use Carp::Assert;

=item B<new>

Constructor

=cut
sub new {
    my $class = shift;
    
    my $self = {};
    bless $self, $class;
    
    return $self;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
