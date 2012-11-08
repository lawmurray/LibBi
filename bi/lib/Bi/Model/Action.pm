=head1 NAME

Bi::Model::Action - action contained within a block.

=head1 SYNOPSIS

    use Bi::Model::Action;
    
    my $action = new Bi::Model::Action(...);

=head1 INHERITS

L<Bi::ArgHandler>

=head1 METHODS

=over 4

=cut

package Bi::Model::Action;

use base 'Bi::ArgHandler';
use warnings;
use strict;

use Carp::Assert;
use Bi::Expression;
use Bi::Utility qw(push_unique);
use Bi::Visitor::GetDims;

=item B<new>(I<id>, I<target>, I<op>, I<expr>)

Factory method for creating actions, but usable as constructor.

=over 4

=item I<id>

Unique numerical id of the action.

=item I<target>

Target to which the action applies, as L<Bi::Expression::VarIdentifier>
object.

=item I<op>

Operator of the action as a string, either '~', '<-' or undef. If undef,
the action is permitted to select its preferred operator.

=item I<expr>

Expression.

=back

Returns the new object, which will be of a class derived from
L<Bi::Model::Action>, not of type L<Bi::Model::Action> directly. The type of
the action is inferred from the expression, in particular if it is wrapped in
a L<Bi::Expression::Function> at the top level, then that function name may
be interpreted as an action name, and the appropriate C<Bi::Action::*> class
instantiated.

=cut
sub new {
    my $class = shift;
    my $id = shift;
    my $target = shift;
    my $op = shift;
    my $expr = shift;

    # pre-conditions
    assert(!defined($target) || $target->isa('Bi::Expression::VarIdentifier')) if DEBUG;
    assert(defined $expr) if DEBUG;
    
    my $args;
    my $named_args;
    my $name;
    
    if ($expr->isa('Bi::Expression::Function') && !$expr->is_math) {
        $name = $expr->get_name;
        $args = $expr->get_args;
        $named_args = $expr->get_named_args;
    } else {
        $name = 'eval_';
        $args = [ $expr ];
        $named_args = {};
    }

    my $self = new Bi::ArgHandler($args, $named_args);
    $self->{_id} = $id;
    $self->{_target} = $target;
    $self->{_dims} = (defined $target) ? $target->get_dims : undef;
    $self->{_op} = $op;
    $self->{_name} = lc($name);
    $self->{_parent} = undef;
    $self->{_is_matrix} = 0;
    $self->{_can_combine} = 0;
    $self->{_is_inplace} = 0;
    $self->{_can_nest} = 0;
    $self->{_unroll_args} = 1;
    $self->{_unroll_target} = 0;
    $self->{_read_vars} = [];
    $self->{_write_vars} = [];
    $self->{_inplace_vars} = [];

    # look up appropriate class name
    $class = "Bi::Action::$name";
    eval ("require $class") || die("don't know what to do with action '$name'\n");

    bless $self, $class;
    $self->validate;
    
    return $self;
}

=item B<new_copy_action>(I<id>, I<to>, I<from>)

Constructor for action that copies one variable to another.

=over 4

=item I<id>

Unique numerical id of the action.

=item I<to>

Variable reference, as L<Bi::Expression::VarIdentifier> object, to which the
action copies.

=item I<from>

Variable reference, as L<Bi::Expression::VarIdentifier> object, from which
the action copies.

=back

Returns the new object.

=cut
sub new_copy_action {
    my $class = shift;
    my $id = shift;
    my $to = shift;
    my $from = shift;
    
    assert ($to->isa('Bi::Expression::VarIdentifier')) if DEBUG;
    assert ($from->isa('Bi::Expression::VarIdentifier')) if DEBUG;
        
    return $class->new($id, $to, '<-', $from);
}

=item B<clone>(I<model>)

Return a clone of the object. A I<model>, of type L<Bi::Model>, is required
to assign a unique id to the action.

=cut
sub clone {
    my $self = shift;
    my $model = shift;
    
    assert (defined($model) && $model->isa('Bi::Model')) if DEBUG;
    
    my $clone = Bi::ArgHandler::clone($self);
    $clone->{_id} = $model->next_action_id;
    $clone->{_target} = $self->get_target->clone;
    $clone->{_dims} = [ @{$self->get_dims} ];
    $clone->{_op} = $self->get_op;
    $clone->{_name} = $self->get_name;
    $clone->{_parent} = $self->get_parent;
    $clone->{_is_matrix} = $self->is_matrix;
    $clone->{_can_combine} = $self->can_combine;
    $clone->{_is_inplace} = $self->is_inplace;
    $clone->{_can_nest} = $self->can_nest;
    $clone->{_unroll_target} = $self->unroll_target;
    $clone->{_unroll_args} = $self->unroll_args;
    $clone->{_read_vars} = [ @{$self->get_read_vars} ];
    $clone->{_write_vars} = [ @{$self->get_write_vars} ];
    $clone->{_inplace_vars} = [ @{$self->get_inplace_vars} ];
    
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

=item B<get_target>

Get the target variable of the action, as a
L<Bi::Expression::VarIdentifier> object.

=cut
sub get_target {
    my $self = shift;
    return $self->{_target};
}

=item B<has_op>

Does the action have an operator defined?

=cut
sub has_op {
    my $self = shift;
    return defined($self->{_op});
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
    
    $self->{_op} = $op;
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

=cut
sub set_name {
    my $self = shift;
    my $name = shift;
    
    return $self->{_name} = $name;
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

=item B<get_dims>

Get dimensions.

=cut
sub get_dims {
    my $self = shift;
    
    return $self->{_dims};
}

=item B<set_dims>(I<dims>)

Set dimensions.

=cut
sub set_dims {
    my $self = shift;
    my $dims= shift;
    
    assert (!defined($dims) || ref($dims) eq 'ARRAY') if DEBUG;

    $self->{_dims} = $dims;
}

=item B<get_read_vars>

Get array ref of variables that are read in the execution of the action.

=cut
sub get_read_vars {
    my $self = shift;
    return $self->{_read_vars};
}

=item B<get_write_vars>

Get array ref of variables that are written in the execution of the action.

=cut
sub get_write_vars {
    my $self = shift;
    return $self->{_write_vars};
}

=item B<get_inplace_vars>

Get array ref of variables that are edited in-place in the execution of
the action.

=cut
sub get_inplace_vars {
    my $self = shift;
    return $self->{_inplace_vars};
}

=item B<get_consts>

Get all constants referenced in the action.

=cut
sub get_consts {
    my $self = shift;
    my $arg;
    my $consts;
    
    foreach $arg (@{$self->get_args}) {
        push_unique($consts, $arg->get_consts);
    }
    foreach $arg (values %{$self->get_named_args}) {
        push_unique($consts, $arg->get_consts);
    }
    return $consts;
}

=item B<get_inlines>

Get all inline expressions referenced in the action.

=cut
sub get_inlines {
    my $self = shift;
    my $arg;
    my $consts;
    
    foreach $arg (@{$self->get_args}) {
        push_unique($consts, $arg->get_inlines);
    }
    foreach $arg (values %{$self->get_named_args}) {
        push_unique($consts, $arg->get_inlines);
    }
    return $consts;
}

=item B<get_vars>

Get all variables referenced in the action.

=cut
sub get_vars {
    my $self = shift;
    my $vars = [];
    my $arg;

    foreach $arg (@{$self->get_args}) {
        push_unique($vars, $arg->get_vars);
    }
    foreach $arg (values %{$self->get_named_args}) {
        push_unique($vars, $arg->get_vars);
    }
    return $vars;
}

=item B<set_target>

Set target variable.

=cut
sub set_target {
    my $self = shift;
    my $target = shift;
    
    assert (!defined($target) || $target->isa('Bi::Expression::VarIdentifier')) if DEBUG;
    
    $self->{_target} = $target;
}

=item B<ensure_op>(I<op>)

Ensure that the action has operator I<op>. If no operator has been assigned,
then assigns I<op>.

=cut
sub ensure_op {
    my $self = shift;
    my $op = shift;
    
    if (!$self->has_op) {
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

Compute intrinsic standard deviation.

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
    warn("action '$name' is missing 'validate' method\n");
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    if (defined $self->{_target}) {
        $self->{_target} = $self->get_target->accept($visitor, @args);
    }
    Bi::ArgHandler::accept($self, $visitor, @args);

    return $visitor->visit($self, @args);
}

1;

=back

=head1 SEE ALSO

L<Bi::Model>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
