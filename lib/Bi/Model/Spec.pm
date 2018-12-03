=head1 NAME

Bi::Model::Spec - block, dimension or variable name with attached
properties.

=head1 SYNOPSIS

    use Bi::Model::Spec;

=head1 INHERITS

L<Bi::ArgHandler>

=head1 DESCRIPTION

I<Bi::Model::Spec> is used to temporarily store a name and associated
properties during parsing. These will be subsumed into a L<Bi::Model>,
L<Bi::Model::Var>, L<Bi::Model::Dim>, L<Bi::Block> or
other object before the completion of the process.

=head1 METHODS

=over 4

=cut

package Bi::Model::Spec;

use parent 'Bi::ArgHandler';
use warnings;
use strict;

use Carp::Assert;
use Scalar::Util 'refaddr';

use Bi::Expression;

=item B<new>(I<name>, I<dims>, I<args>, I<named_args>)

Constructor.

=over 4

=item I<name>

Name of the block, variable or dimension.

=item I<dims>

Ordered list of dimensions, as L<Bi::Model::Dim> objects.

=item I<args>

Ordered list of positional arguments, as L<Bi::Expression> objects.

=item I<named_args>

Hash of named arguments, keyed by name, as L<Bi::Expression> objects.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $name = shift;
    my $dims = shift;
    my $args = shift;
    my $named_args = shift;

    # pre-conditions
    assert(!defined($dims) || ref($dims) eq 'ARRAY') if DEBUG;    
    assert(!defined($args) || ref($args) eq 'ARRAY') if DEBUG;    
    assert(!defined($named_args) || ref($named_args) eq 'HASH') if DEBUG;    
    map { assert($_->isa('Bi::Model::Dim')) if DEBUG } @$dims;

    my $self = new Bi::ArgHandler($args, $named_args);
    $self->{_name} = $name;
    $self->{_dims} = $dims;

    bless $self, $class;
   
    return $self;
}

=item B<get_name>

Get name.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<get_dims>

Get dimensions.

=cut
sub get_dims {
    my $self = shift;
    return $self->{_dims};
}

1;

=back

=head1 SEE ALSO

L<Bi::Model>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

