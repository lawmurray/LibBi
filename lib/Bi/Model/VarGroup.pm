=head1 NAME

Bi::Model::VarGroup - variable group.

=head1 SYNOPSIS

    use Bi::Model::VarGroup;
    
    my $group = new Bi::Model::VarGroup($type, $name);
    $group->push_var($var1);
    $group->push_var($var2);
    $group->push_var($var3);
    $model->push_var_group($group);

=head1 INHERITS

L<Bi::ArgHandler>

=cut

package Bi::Model::VarGroup;

use parent 'Bi::Node';
use warnings;
use strict;

use Carp::Assert;
use Bi::Model::Dim;

=head1 METHODS

=over 4

=item B<new>(I<type>, I<name>)

Constructor.

=over 4

=item I<type>

Type of variables in the group.

=item I<name>

Name of the group.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $type = shift;
    my $name = shift;

    my $self = {
        _type => $type,
        _name => $name,
        _vars => []
    };
    bless $self, $class;
    
    return $self;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;
    
    my $clone = {
        _type => $self->get_type,
        _name => $self->get_name,
        _vars => $self->get_vars
    };
    bless $clone, ref($self);
    return $clone;
}

=item B<get_type>

Get the type of variables in the group.

=cut
sub get_type {
    my $self = shift;
    return $self->{_type};
}

=item B<set_type>(I<type>)

Set the type of variables in the group. 

=cut
sub set_type {
    my $self = shift;
    my $type = shift;
    
    $self->{_type} = $type;
}

=item B<get_name>

Get the name of the group.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<set_name>(I<name>)

Set the name of the group.

=cut
sub set_name {
    my $self = shift;
    my $name = shift;
    
    $self->{_name} = $name;
}

=item B<get_vars>

Get all variables declared in the block.

=cut
sub get_vars {
    my $self = shift;
    
    return $self->{_vars};
}

=item B<get_var>(I<name>)

Get the variable called I<name>, or undef if it does not exist.

=cut
sub get_var {
    my $self = shift;
    my $name = shift;

    return $self->_get_item($self->get_vars, $name);
}

=item B<is_var>(I<name>)

Is there a variable called I<name>?

=cut
sub is_var {
    my $self = shift;
    my $name = shift;
    
    return $self->_is_item($self->get_vars, $name);
}

=item B<push_var>(I<var>)

Add a variable.

=cut
sub push_var {
    my $self = shift;
    my $var = shift;
    
    assert ($var->isa('Bi::Model::Var')) if DEBUG;
    assert ($var->get_type eq $self->get_type) if DEBUG;
    
    push(@{$self->get_vars}, $var);
}

=item B<get_size>

Get the size of the variable group (sum of the sizes of all variables in the
group).

=cut
sub get_size {
    my $self = shift;
    
    my $size = 0;
    foreach my $var (@{$self->get_vars}) {
        $size += $var->get_size;
    }
    return $size;
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    $self = $visitor->visit_before($self, @args);
    for (my $i = 0; $i < @{$self->get_vars}; ++$i) {
    	$self->get_vars->[$i] = $self->get_vars->[$i]->accept($visitor, @args);
    }
    return $visitor->visit_after($self, @args);
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
