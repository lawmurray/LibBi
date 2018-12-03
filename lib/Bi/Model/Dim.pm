=head1 NAME

Bi::Model::Dim - dimension.

=head1 SYNOPSIS

    use Bi::Model::Dim;

=head1 INHERITS

L<Bi::ArgHandler>

=head1 METHODS

=over 4

=cut

package Bi::Model::Dim;

use parent 'Bi::Node', 'Bi::ArgHandler';
use warnings;
use strict;

use Carp::Assert;
use Scalar::Util 'refaddr';

our $_next_dim_id = 0;

our $DIM_ARGS = [
  {
    name => 'size',
    positional => 1,
    mandatory => 1
  },
  {
    name => 'boundary',
    positional => 1,
    default => 'none'
  }
];

=item B<new>(I<name>, I<args>, I<named_args>)

Constructor.

=over 4

=item I<name>

Name of the dimension. If undefined a unique name is generated.

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
    my $args = shift;
    my $named_args = shift;

    my $id = $_next_dim_id++;
    my $self = new Bi::ArgHandler($args, $named_args);
    $self->{_id} = $id;
    $self->{_name} = (defined $name) ? $name : sprintf("dim_%d_", $id);

    bless $self, $class;

    $self->validate;
    return $self;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;

    my $clone = Bi::ArgHandler::clone($self);
    $clone->{_id} = $_next_dim_id++;
    $clone->{_name} = $self->get_name;
    
    bless $clone, ref($self);
    return $clone;
}

=item B<get_id>

Get the id of the dimension.

=cut
sub get_id {
    my $self = shift;
    return $self->{_id};
}

=item B<get_name>

Get the name of the dimension.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<get_size>

Get the size of the dimension.

=cut
sub get_size {
    my $self = shift;
    return $self->get_named_arg('size')->eval_const;
}

=item B<gen_index>

Generate index for this dimension, for use in
L<Bi::Expression::VarIdentifier> objects.

=cut
sub gen_index {
    my $self = shift;

    return new Bi::Expression::Index(new Bi::Expression::DimAliasIdentifier($self));
}

=item B<gen_range>

Generate range for this dimension, for use in
L<Bi::Expression::VarIdentifier> objects.

=cut
sub gen_range {
	my $self = shift;
	
	return new Bi::Expression::Range(new Bi::Expression::IntegerLiteral(0), new Bi::Expression::IntegerLiteral($self->get_size - 1));
}

=item B<gen_alias>

Generate alias for this dimension, for use in L<Bi::Action> objects.

=cut
sub gen_alias {
	my $self = shift;
	
	return new Bi::Model::DimAlias(undef, new Bi::Expression::Range(new Bi::Expression::IntegerLiteral(0), new Bi::Expression::IntegerLiteral($self->get_size - 1)));
}

=item B<validate>

Validate dimension.

=cut
sub validate {
    my $self = shift;
    
    $self->process_args($DIM_ARGS);
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    my $new = $visitor->visit_before($self, @args);
    if (refaddr($new) == refaddr($self)) {
    	Bi::ArgHandler::accept($self, $visitor, @args);
    	$new = $visitor->visit_after($self, @args);
    }
    return $new;
}

=item B<equals>(I<obj>)

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return ref($obj) eq ref($self) && $self->get_id == $obj->get_id;
}

1;

=back

=head1 SEE ALSO

L<Bi::Model>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

