=head1 NAME

Bi::Model::Var - variable.

=head1 SYNOPSIS

    use Bi::Model::Var;

=head1 INHERITS

L<Bi::ArgHandler>

=head1 METHODS

=over 4

=cut

package Bi::Model::Var;

use base 'Bi::ArgHandler';
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

=item C<input_name> (default the same as the name of the variable)

Name to use for the variable in input files.

=item C<output_name> (default the same as the name of the variable)

Name to use for the variable in output files.

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
        name => 'input_name',
    },
    {
        name => 'output_name',
    }
];

=item B<new>(I<name>, I<dims>, I<args>, I<named_args>)

Constructor.

=over 4

=item I<name>

Unique name of the variable.

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
    my $name = shift;
    my $dims = shift;
    my $args = shift;
    my $named_args = shift;

    # pre-conditions
    assert(!defined($dims) || ref($dims) eq 'ARRAY') if DEBUG;
    map { assert($_->isa('Bi::Model::Dim')) if DEBUG } @$dims;

    my $self = new Bi::ArgHandler($args, $named_args);    
    $self->{_id} = -1;
    $self->{_name} = $name;
    $self->{_type} = undef; # supplied by derived class
    $self->{_dims} = $dims;
    
    bless $self, $class;
   
    $self->validate;
   
    return $self;
}

=item B<get_id>

Get the id of the variable (-1 until assigned to a L<Bi::Model>).

=cut
sub get_id {
    my $self = shift;
    return $self->{_id};
}

=item B<set_id>(I<id>)

St the id of the variable.

=cut
sub set_id {
    my $self = shift;
    my $id = shift;
    $self->{_id} = $id;
}

=item B<get_name>

Get the name of the variable.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<get_type>

Type of the variable as a string, e.g. 'state', 'noise', 'input', 'obs',
'param'. 

=cut
sub get_type {
    my $self = shift;
    return $self->{_type};
}

=item B<get_dims>

Get the array ref of dimensions associated with the variable, as
L<Bi::Model::Dim> objects.

=cut
sub get_dims {
    my $self = shift;
    return $self->{_dims};
}

=item B<num_dims>

Number of dimensions associated with the variable.

=cut
sub num_dims {
    my $self = shift;    
      return scalar(@{$self->get_dims});
}

=item B<get_size>

Get the size of the variable (product of the sizes of all dimensions along
which it is defined).

=cut
sub get_size {
    my $self = shift;
    
    my $size = 1;
    my $dim;
    foreach $dim (@{$self->get_dims}) {
        $size *= $dim->get_size;
    }
    
    return $size;
}

=item B<gen_aliases>

Generate list of aliases for this variable's dimensions, for use in 
L<Bi::Model::Target> objects.

=cut
sub gen_aliases {
	my $self = shift;
	
	my @aliases;
	my $i;
	for ($i = 0; $i < $self->num_dims; ++$i) {
		my $dim = $self->get_dims->[$i];
		my $name = $dim->get_name . $i . '_';
		my $alias = new Bi::Model::DimAlias($name);
		push(@aliases, $alias);
	}
	return \@aliases;
}

=item B<gen_indexes>

Generate list of indexes for this variable, for use in
L<Bi::Expression::VarIdentifier> objects.

=cut
sub gen_indexes {
	my $self = shift;
	
	my $aliases = $self->gen_aliases;
	my @indexes = map { new Bi::Expression::DimAliasIdentifier($_) } @$aliases;
	
	return \@indexes;
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
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    #Bi::ArgHandler::accept($self, $visitor);
    return $visitor->visit($self, @args);
}

=item B<equals>(I<obj>)

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return ref($obj) eq ref($self) && $self->get_name eq $obj->get_name;
}

1;

=back

=head1 SEE ALSO

L<Bi::Model>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
