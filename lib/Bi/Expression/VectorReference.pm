=head1 NAME

Bi::Expression::VectorReference - vector of expressions.

=head1 INHERITS

L<Bi::Matrix>

=head1 METHODS

=over 4

=cut

package Bi::Expression::VectorReference;

use parent 'Bi::Expression::MatrixReference';
use warnings;
use strict;

use Carp::Assert;

=item B<new>(I<size>)

Constructor.

=cut
sub new {
    my $class = shift;
    my $data = shift;
    my $start = shift;
    my $size = shift;
    
    assert ($size >= 0) if DEBUG;

    my $self = new Bi::Expression::MatrixReference($data, $start, $size, 0, 1, $size);
    bless $self, $class;
    
    return $self;
}

=item B<start>

Starting index in underlying data.

=cut
sub start {
    my $self = shift;
    return $self->start1;
}

=item B<size>

Number of elements.

=cut
sub size {
    my $self = shift;
    return $self->size1;
}

=item B<get>(I<i>)

Get an element.

=cut
sub get {
    my $self = shift;
    my $i = shift;

    return $self->get($i, 0);
}

=item B<set>(I<i>, I<val>)

Set an element.

=cut
sub set {
    my $self = shift;
    my $i = shift;
    my $val = shift;

    $self->set($i, 0, $val);
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev: 3738 $ $Date: 2013-04-16 23:24:15 +1000 (Tue, 16 Apr 2013) $
