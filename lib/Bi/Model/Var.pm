=head1 NAME

Bi::Model::Var - variable.

=head1 SYNOPSIS

    use Bi::Model::Var;

=head1 INHERITS

L<Bi::ArgHandler>

=cut

package Bi::Model::Var;

use parent 'Bi::Node', 'Bi::ArgHandler';
use warnings;
use strict;

use Carp::Assert;
use Bi::Model::Dim;

=head1 PARAMETERS

=over 4

=item C<has_input> (default 1)

Include variable when doing input from a file?

=item C<has_output> (default 1)

Include variable when doing output to a file?

=item C<input_name> (default the same as variable name)

Name to use for the variable in input files.

=item C<output_name> (default the same as variable name)

Name to use for the variable in output files.

=item C<output_once> (default according to variable type)

Output the variable only once, not at each time.

=back

=cut
our $VAR_ARGS = [
    {
        name => 'has_input',
        default => 1
    },
    {
        name => 'has_output',
        default => 1
    },
    {
        name => 'input_name'
    },
    {
        name => 'output_name'
    },
    {
        name => 'output_once'
    }
];

our $_next_var_id = 0;

our %_types = (
    'input' => 0,
    'noise' => 0,
    'obs' => 0,
    'param' => 0,
    'param_aux_' => 0,
    'state' => 0,
    'state_aux_' => 0
);

=head1 METHODS

=over 4

=item B<new>(I<type>, I<name>, I<dims>, I<args>, I<named_args>)

Constructor.

=over 4

=item I<type>

Type of the variable (e.g. 'input', 'state', 'param').

=item I<name>

Name of the variable. If undefined, a unique name is generated.

=item I<dims>

Ordered list of the dimensions associated with the variable, as
L<Bi::Model::Dim> objects.

=item I<args>

Ordered list of positional arguments, as L<Bi::Expression> objects.

=item I<named_args>

Hash of named arguments, keyed by name, as L<Bi::Expression> objects.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $type = shift;
    my $name = shift;
    my $dims = shift;
    my $args = shift;
    my $named_args = shift;

    # pre-conditions
    assert(exists $_types{$type}) if DEBUG;
    assert(!defined($dims) || ref($dims) eq 'ARRAY') if DEBUG;
    map { assert($_->isa('Bi::Model::Dim')) if DEBUG } @$dims;

    my $id = $_next_var_id++;
    my $self = new Bi::ArgHandler($args, $named_args);    
    $self->{_id} = $id;
    $self->{_type} = $type;
    $self->{_name} = (defined $name) ? $name : sprintf("${type}_%d_", $id),
    $self->{_dims} = $dims;
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
    $clone->{_id} = $_next_var_id++;
    $clone->{_type} = $self->get_type;
    $clone->{_name} = $self->get_name;
    $clone->{_dims} = $self->get_dims;
    
    bless $clone, ref($self);
    return $clone;
}

=item B<get_id>

Get the id of the variable.

=cut
sub get_id {
    my $self = shift;
    return $self->{_id};
}

=item B<get_type>

Get the type of the variable.

=cut
sub get_type {
    my $self = shift;
    return $self->{_type};
}

=item B<set_type>(I<type>)

Set the type of the variable. 

=cut
sub set_type {
    my $self = shift;
    my $type = shift;
        
    assert (exists $_types{$type}) if DEBUG;
    
    $self->{_type} = $type;
}

=item B<get_name>

Get the name of the variable.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<set_name>(I<name>)

Set the name of the variable.

=cut
sub set_name {
    my $self = shift;
    my $name = shift;
    
    $self->{_name} = $name;
}

=item B<get_dims>

Get the array ref of dimensions associated with the variable, as
L<Bi::Model::Dim> objects.

=cut
sub get_dims {
    my $self = shift;
    return $self->{_dims};
}

=item B<get_shape>

Get the shape of the variable, as an array ref of sizes.

=cut
sub get_shape {
    my $self = shift;
    
    my @shape = map { $_->get_size } @{$self->get_dims};
    return \@shape;
}

=item B<get_size>

Get the size of the variable (product of the sizes of all dimensions along
which it is defined).

=cut
sub get_size {
    my $self = shift;
    
    my $size = 1;
    foreach my $len (@{$self->get_shape}) {
        $size *= $len;
    }
    
    return $size;
}

=item B<gen_indexes>

Generate array ref of indexes for this variable, for use in
L<Bi::Expression::VarIdentifier> objects.

=cut
sub gen_indexes {
	my $self = shift;
	
	my @indexes = map { new Bi::Expression::Index(new Bi::Expression::DimAliasIdentifier($_)) } @{$self->gen_aliases};
	
	return \@indexes;
}

=item B<gen_ranges>

Generate array ref of ranges for this variable, for use in
L<Bi::Expression::VarIdentifier> objects.

=cut
sub gen_ranges {
	my $self = shift;
	
	my @ranges = map { $_->gen_range } @{$self->get_dims};
	
	return \@ranges;
}

=item B<gen_aliases>

Generate array ref of dimension aliases for this variable, for use in
L<Bi::Expression::VarIdentifier> objects.

=cut
sub gen_aliases {
	my $self = shift;
	
	my @aliases = map { $_->gen_alias } @{$self->get_dims};
	
	return \@aliases;
}

=item B<validate>

Validate variable.

=cut
sub validate {
    my $self = shift;
    
    $self->process_args($VAR_ARGS);
    
    # apply defaults that process_args can't handle
    if (!$self->is_named_arg('input_name')) {
        $self->set_named_arg('input_name', new Bi::Expression::StringLiteral($self->get_name));
    }
    if (!$self->is_named_arg('output_name')) {
        $self->set_named_arg('output_name', new Bi::Expression::StringLiteral($self->get_name));
    }
    if (!$self->is_named_arg('output_once')) {
        my $once = int($self->get_type eq 'param' || $self->get_type eq 'param_aux_');
        $self->set_named_arg('output_once', new Bi::Expression::Literal($once));
    }
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    $self = $visitor->visit_before($self, @args);
    Bi::ArgHandler::accept($self, $visitor);
    
    return $visitor->visit_after($self, @args);
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

=head1 VERSION

$Rev$ $Date$
