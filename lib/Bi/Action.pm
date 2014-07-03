=head1 NAME

Bi::Action - action contained within a block.

=head1 SYNOPSIS

    use Bi::Action;
    
    my $action = new Bi::Action;
    $action->set_left(...);
    $action->set_op(...);
    $action->set_right(...);
    $action->validate;

=head1 INHERITS

L<Bi::ArgHandler>

=head1 METHODS

=over 4

=cut

package Bi::Action;

use parent 'Bi::Node', 'Bi::ArgHandler';
use warnings;
use strict;

use Carp::Assert;
use Bi::Model::DimAlias;
use Bi::Expression;
use Bi::Utility qw(push_unique);
use Bi::Visitor::GetDims;

our $_next_action_id = 0;

=item B<new>

Constructor.

=cut
sub new {
    my $class = shift;

    my $self = new Bi::ArgHandler;
    $self->{_id} = $_next_action_id++;
    $self->{_aliases} = [];
    $self->{_left} = undef;
    $self->{_op} = undef;
    $self->{_name} = undef;
    $self->{_shape} = new Bi::Expression::Shape();
    $self->{_parent} = undef;
    $self->{_is_matrix} = 0;
    $self->{_can_combine} = 0;
    $self->{_is_inplace} = 0;
    $self->{_can_nest} = 0;
    $self->{_unroll_args} = 1;
    $self->{_unroll_target} = 0;

    bless $self, $class;    
    return $self;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;
    
    my $clone = Bi::ArgHandler::clone($self);
    $clone->{_id} = $_next_action_id++;
    $clone->{_aliases} = $self->get_aliases; # don't clone
    $clone->{_left} = $self->get_left->clone;
    $clone->{_op} = $self->get_op;
    $clone->{_name} = $self->get_name;
    $clone->{_shape} = $self->get_shape->clone;
    $clone->{_parent} = $self->get_parent;
    $clone->{_is_matrix} = $self->is_matrix;
    $clone->{_can_combine} = $self->can_combine;
    $clone->{_is_inplace} = $self->is_inplace;
    $clone->{_can_nest} = $self->can_nest;
    $clone->{_unroll_args} = $self->unroll_args;
    $clone->{_unroll_target} = $self->unroll_target;
    
    bless $clone, ref($self);
    return $clone; 
}

=item B<get_id>

Get the id of the action.

=cut
sub get_id {
    my $self = shift;
    return $self->{_id};
}

=item B<get_type>

Returns the string "action".

=cut
sub get_type {
    return "action";
}

=item B<get_aliases>

Get dimension aliases declared in the action, as array ref of
L<Bi::Model::DimAlias> objects.


=cut
sub get_aliases {
    my $self = shift;
    return $self->{_aliases};
}

=item B<get_alias>(I<name>)

Get the dimension alias called I<name>, or undef if it does not exist.

=cut
sub get_alias {
    my $self = shift;
    my $name = shift;

    return $self->_get_item($self->get_aliases, $name);
}

=item B<set_aliases>(I<aliases>)

Set dimension aliases.

=cut
sub set_aliases {
    my $self = shift;
    my $aliases = shift;
    
    assert(ref($aliases) eq 'ARRAY') if DEBUG;
    map { assert($_->isa('Bi::Model::DimAlias')) } @$aliases if DEBUG;

    $self->{_aliases} = $aliases;
}

=item B<get_left>

Get the left side of the action, as a L<Bi::Model::VarIdentifier> object.

=cut
sub get_left {
    my $self = shift;
    return $self->{_left};
}

=item B<set_left>(I<left>)

Set the left side of the action.

=cut
sub set_left {
    my $self = shift;
    my $left = shift;
    
    assert ($left->isa('Bi::Expression::VarIdentifier')) if DEBUG;
        
    # set default aliases and ranges if they haven't been set already
    if (@{$self->get_aliases} == 0) {
    	$self->set_aliases($left->get_var->gen_aliases);
    	$left->set_indexes($left->get_var->gen_ranges);
    }
    $self->{_left} = $left;
}

=item B<get_op>

Get the operator of the action.

=cut
sub get_op {
    my $self = shift;
    return $self->{_op};
}

=item B<set_op>(I<op>)

Set the operator of the action.

=cut
sub set_op {
    my $self = shift;
    my $op = shift;
    
    assert ($op eq '<-' || $op eq '~' || $op eq '=') if DEBUG;
    
    $self->{_op} = $op;
}

=item B<set_right>(I<right>)

Set the right side of the action. Note that there is no matching B<get_right>
function, as the right-hand side is always further parsed into a name and
arguments.

=cut
sub set_right {
    my $self = shift;
    my $right = shift;
 
    my $name;
    my $args;
    my $named_args;
 
    if ($right->isa('Bi::Expression::Function') && $right->is_action) {
        $name = $right->get_name;
        $args = $right->get_args;
        $named_args = $right->get_named_args;
    } else {
        my $op = $self->get_op;
        if ($op eq '<-') {
            $name = 'eval_';
        } elsif ($op eq '~') {
            $name = 'pdf';
        } elsif ($op eq '=') {
            $name = 'ode_';
        } else {
            assert 0 if DEBUG;
        }
        $args = [ $right ];
        $named_args = {};
    }

    $self->set_name($name);
    $self->set_args($args);
    $self->set_named_args($named_args);
}

=item B<get_name>

Get the name of the action.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<set_name>(I<name>)

Set the name of the action.

A change in name will morph the object into a new type derived from
L<Bi::Action>.

=cut
sub set_name {
    my $self = shift;
    my $name = shift;
    
    # morph to appropriate class for name
    my $class = 'Bi::Action';
	if (defined $name) {
	    if ($name !~ /^\w+$/) {
	    	die("don't know what to do with action '$name'\n");
	    } else {
	    	$class = "Bi::Action::$name";
	        eval("require $class") || die("don't know what to do with action '$name'\n");
	    }
	}
    bless $self, $class;
    
    $self->{_name} = $name;
}

=item B<get_right_var_refs>

Get all variable references that appear on the right side of the action, as
array ref of B<Bi::Expression::VarIdentifier> objects.

=cut
sub get_right_var_refs {
    my $self = shift;
    
    my $visitor = new Bi::Visitor::GetNodesOfType;
    my @refs;
    Bi::ArgHandler::accept($self, $visitor, [ 'Bi::Expression::VarIdentifier' ], \@refs);

    return \@refs;
}

=item B<get_right_vars>

Get all variables that appear on the right side of the action, as array ref
of B<Bi::Model::Var> objects. This is used internally to determine
dependencies between actions.

=cut
sub get_right_vars {
    my $self = shift;
    
    my @vars = map { $_->get_var } @{$self->get_right_var_refs};
    
    return \@vars;
}

=item B<get_size>

Get the size of the action.

=cut
sub get_size {
    my $self = shift;
    
    my $size = 1;
    foreach my $alias (@{$self->get_aliases}) {
        $size *= $alias->get_size;
    }
    return $size;
}

=item B<get_shape>

Get the shape of the expression result, as an array ref of sizes.

=cut
sub get_shape {
    my $self = shift;
    
    return $self->{_shape};
}

=item B<set_shape>(I<shape>)

Set shape.

=cut
sub set_shape {
    my $self = shift;
    my $shape = shift;
    
    assert (ref($shape) eq 'Bi::Expression::Shape') if DEBUG;

    $self->{_shape} = $shape;
}

=item B<get_parent>

Get the name of the required parent block type.

=cut
sub get_parent {
    my $self = shift;
    return $self->{_parent};
}

=item B<set_parent>(I<parent>)

Set the name of the required parent block type.

=cut
sub set_parent {
    my $self = shift;
    my $parent = shift;
    $self->{_parent} = $parent;
}

=item B<can_combine>

Can the action be combined with sibling actions that have the same parent
block type?

=cut
sub can_combine {
    my $self = shift;
    return $self->{_can_combine};
}

=item B<set_can_combine>(I<on>)

Can the action be combined with sibling actions that have the same parent
block type?

=cut
sub set_can_combine {
    my $self = shift;
    my $on = shift;
    $self->{_can_combine} = $on;
}

=item B<is_matrix>

Is the action a matrix operation?

=cut
sub is_matrix {
    my $self = shift;
    return $self->{_is_matrix};
}

=item B<set_is_matrix>(I<on>)

Is the action a matrix operation?

=cut
sub set_is_matrix {
    my $self = shift;
    my $on = shift;
    $self->{_is_matrix} = $on;
}

=item B<unroll_target>

Should the target of the action be unrolled?

=cut
sub unroll_target {
    my $self = shift;
    return $self->{_unroll_target};
}

=item B<set_unroll_target>(I<on>)

Should the target of the action be unrolled?

=cut
sub set_unroll_target {
    my $self = shift;
    my $on = shift;
    $self->{_unroll_target} = $on;
}

=item B<unroll_args>

Should the arguments of the action be unrolled?

=cut
sub unroll_args {
    my $self = shift;
    return $self->{_unroll_args};
}

=item B<set_unroll_args>(I<on>)

Should the arguments of the action be unrolled?

=cut
sub set_unroll_args {
    my $self = shift;
    my $on = shift;
    $self->{_unroll_args} = $on;
}

=item B<is_inplace>

Does the action perform in-place edits?

=cut
sub is_inplace {
    my $self = shift;
    return $self->{_is_inplace};
}

=item B<set_is_inplace>(I<on>)

Does the action perform in-place edits?

=cut
sub set_is_inplace {
    my $self = shift;
    my $on = shift;
    $self->{_is_inplace} = $on;
}

=item B<can_nest>

Can the action be nested within other actions?

=cut
sub can_nest {
    my $self = shift;
    return $self->{_can_nest};
}

=item B<set_can_nest>(I<on>)

Can the action be nested within other actions? This should only be the case
for actions which can compute the size of their result from the sizes of
their arguments. Actions which rely on context for size cannot be nested.

=cut
sub set_can_nest {
    my $self = shift;
    my $on = shift;
    $self->{_can_nest} = $on;
}

=item B<ensure_op>(I<op>)

Ensure that the action has operator I<op>. If no operator has been assigned,
then assigns I<op>.

=cut
sub ensure_op {
    my $self = shift;
    my $op = shift;
    
    if (!defined $self->get_op) {
        $self->set_op($op);
    } else {
        if ($self->get_op ne $op) {
            my $action = $self->get_name;
            die("action '$action' may only be used with the '$op' operator\n");
        }
    }
}

=item B<mean>

Compute mean.

=cut
sub mean {
    my $self = shift;
    my $name = $self->get_name;
    warn("action '$name' is missing the 'mean' method\n");
    return undef;
}

=item B<std>

Compute standard deviation.

=cut
sub std {
    my $self = shift;
    my $name = $self->get_name;
    warn("action '$name' is missing the 'std' method\n");
    return undef;
}

=item B<jacobian>

Compute partial derivatives.

=cut
sub jacobian {
    my $self = shift;
    my $name = $self->get_name;
    warn("action '$name' is missing the 'jacobian' method\n");
    return ([], []);
}

=item B<validate>

Validate arguments.

=cut
sub validate {
    my $self = shift;

    my $name = $self->get_name;
    my $num_aliases = @{$self->get_aliases};
    
    # explicitly set alias ranges
    for (my $i = 0; $i < $num_aliases; ++$i) {
        my $alias = $self->get_aliases->[$i];
        my $dim = $self->get_left->get_var->get_dims->[$i];
        
        if (!$alias->has_range) {
            my $range = new Bi::Expression::Range(new Bi::Expression::IntegerLiteral(0), new Bi::Expression::IntegerLiteral($dim->get_size - 1));
            $alias->set_range($range);
        }
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
    Bi::ArgHandler::accept($self, $visitor, @args);
    $self->{_left} = $self->get_left->accept($visitor, @args);
    for (my $i = 0; $i < @{$self->get_aliases}; ++$i) {
    	$self->get_aliases->[$i] = $self->get_aliases->[$i]->accept($visitor, @args);
    }
    return $visitor->visit_after($self, @args);
}

=item B<equals>(I<obj>)

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return ref($obj) eq ref($self) && $self->get_id == $obj->get_id;
}

=item B<_get_item>(I<list>, I<name>)

Get the item called I<name> from I<list>, or undef if it does not exist.

=cut
sub _get_item {
    my $self = shift;
    my $list = shift;
    my $name = shift;
    
    foreach my $item (@$list) {
        if (defined $item->get_name && $item->get_name eq $name) {
            return $item;
        }
    }
    return undef;
}

1;

=back

=head1 SEE ALSO

L<Bi::Model>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
