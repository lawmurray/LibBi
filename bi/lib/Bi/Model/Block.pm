=head1 NAME

Bi::Model::Block - block.

=head1 SYNOPSIS

    use Bi::Model::Block;

=head1 INHERITS

L<Bi::ArgHandler>

=head1 METHODS

=over 4

=cut

package Bi::Model::Block;

use base 'Bi::ArgHandler';
use warnings;
use strict;

use Carp::Assert;
use Bi::Model::Action;
use Bi::Visitor::CanSimulate;

=item B<new>(I<id>, I<name>, I<args>, I<named_args>, I<actions>, I<blocks>)

Factory method for creating blocks, but usable as constructor.

=over 4

=item I<id>

Unique numerical id of the block.

=item I<name>

Name of the block, if any.

=item I<args>

Ordered list of positional arguments as L<Bi::Expression> objects.

=item I<named_args>

Hash of named arguments, keyed by name, as L<Bi::Expression> objects.

=item I<actions>

Ordered list of actions contained by the block as L<Bi::Model::Action>
objects.

=item I<blocks>

Ordered list of sub-blocks contained by the block as L<Bi::Model::Block>
objects.

=back

Returns the new object, which will be of a class derived from
L<Bi::Model::Block>, not of type L<Bi::Model::Block> directly. The type of
the block is inferred from the block's name, and the appropriate
C<Bi::Block::*> class instantiated.

=cut
sub new {
    my $class = shift;
    my $id = shift;
    my $name = shift;
    my $args = scalar(@_) ? shift : [];
    my $named_args = scalar(@_) ? shift : {};
    my $actions = scalar(@_) ? shift : [];
    my $blocks = scalar(@_) ? shift : [];

    # pre-condition
    assert(!defined($args) || ref($args) eq 'ARRAY') if DEBUG;
    assert(!defined($named_args) || ref($named_args) eq 'HASH') if DEBUG;
    assert(!defined($actions) || ref($actions) eq 'ARRAY') if DEBUG;
    assert(!defined($blocks) || ref($blocks) eq 'ARRAY') if DEBUG;
    map { assert($_->isa('Bi::Model::Action')) if DEBUG } @$actions;
    map { assert($_->isa('Bi::Model::Block')) if DEBUG } @$blocks;

    $name = 'eval' unless defined($name); 
    $args = [] unless defined($args);
    $named_args = {} unless defined($named_args);
    $actions = [] unless defined($actions);
    $blocks = [] unless defined($blocks);

    my $self = new Bi::ArgHandler($args, $named_args);
    $self->{_id} = $id;
    $self->{_name} = lc($name);
    $self->{_commit} = 0;
    $self->{_actions} = $actions;
    $self->{_blocks} = $blocks;

    # look up appropriate class name
    $class = "Bi::Block::$name";
    eval ("require $class") || die("don't know what to do with block '$name'\n");

    bless $self, $class;
    $self->validate;
    
    return $self;
}

=item B<clone>(I<model>)

Return a clone of the object. A I<model>, of type L<Bi::Model>, is required
to assign a unique id to the block, and its sub-blocks and actions.

=cut
sub clone {
    my $self = shift;
    my $model = shift;
    
    assert (defined($model) && $model->isa('Bi::Model')) if DEBUG;
    
    my $clone = Bi::ArgHandler::clone($self);
    $clone->{_id} = $model->next_block_id;
    $clone->{_name} = $self->get_name;
    $clone->{_commit} = $self->get_commit;
    $clone->{_actions} = [ map { $_->clone($model) } @{$self->get_actions} ];
    $clone->{_blocks} = [ map { $_->clone($model) } @{$self->get_blocks} ];
    
    bless $clone, ref($self);
    
    return $clone; 
}

=item B<clear>

Remove all subblocks and actions.

=cut
sub clear {
    my $self = shift;
    
    $self->{_actions} = [];
    $self->{_blocks} = [];
}

=item B<new_copy_block>(I<id>)

Constructor for block of copy actions.

=cut
sub new_copy_block {
    my $class = shift;
    my $id = shift;
    
    return $class->new($id, 'eval');
}

=item B<get_id>

Get the id of the block.

=cut
sub get_id {
    my $self = shift;
    return $self->{_id};
}

=item B<get_name>

Get the name of the block.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<set_id>(I<id>)

Set the id of the block.

=cut
sub set_id {
    my $self = shift;
    my $id = shift;
    $self->{_id} = $id;
}

=item B<set_name>(I<name>)

Set the name of the block.

=cut
sub set_name {
    my $self = shift;
    my $name = shift;
    $self->{_name} = $name;
}

=item B<get_commit>

Should writes be committed after execution of the block?

=cut
sub get_commit {
    my $self = shift;
    return $self->{_commit};
}

=item B<set_commit>(I<on>)

Should writes be committed after execution of the block?

=cut
sub set_commit {
    my $self = shift;
    my $on = shift;
    $self->{_commit} = $on;
}

=item B<get_actions>

Get ordered list of actions contained by the block as L<Bi::Model::Action>
objects.

=cut
sub get_actions {
    my $self = shift;
    return $self->{_actions};
}

=item B<set_actions>(I<actions>)

Set actions.

=cut
sub set_actions {
    my $self = shift;
    my $actions = shift;
    
    # pre-conditions
    assert(ref($actions) eq 'ARRAY') if DEBUG;
    map { assert($_->isa('Bi::Model::Action')) if DEBUG } @$actions;
    
    $self->{_actions} = $actions;
}

=item B<get_blocks>

Get ordered list of sub-blocks contained by the block as L<Bi::Model::Block>
objects.

=cut
sub get_blocks {
    my $self = shift;
    return $self->{_blocks};
}

=item B<set_blocks>(I<blocks>)

Set blocks.

=cut
sub set_blocks {
    my $self = shift;
    my $blocks = shift;
    
    # pre-conditions
    assert(ref($blocks) eq 'ARRAY') if DEBUG;
    map { assert($_->isa('Bi::Model::Block')) if DEBUG } @$blocks;
    
    $self->{_blocks} = $blocks;
}

=item B<get_action>(I<i>)

Get single action.

=over 4

=item * I<i> Action index. Defaults to 0.,

=back

Returns the action.

=cut
sub get_action {
    my $self = shift;
    my $i = (@_) ? shift : 0;
    
    # pre-condition
    assert ($i >= 0 && $i < $self->num_actions) if DEBUG;
    
    return $self->{_actions}->[$i];
}

=item B<get_block>(I<i>)

Get single sub-block.

=over 4

=item * I<i> Sub-block index. Defaults to 0.,

=back

Returns the sub-block.

=cut
sub get_block {
    my $self = shift;
    my $i = (@_) ? shift : 0;
    
    return $self->get_blocks->[$i];
}

=item B<num_actions>

Get number of actions contained by the block.

=cut
sub num_actions {
    my $self = shift;
    return scalar(@{$self->get_actions});
}

=item B<num_blocks>

Get number of sub-blocks contained by the block.

=cut
sub num_blocks {
    my $self = shift;
    
    return scalar(@{$self->get_blocks});
}

=item B<get_consts>

Get all constants referenced in actions of the block.

=cut
sub get_consts {
    my $self = shift;
    my $action;
    my $consts;
    
    foreach $action (@{$self->get_actions}) {
      Bi::Utility::push_unique($consts, $action->get_consts);
    }
    return $consts;
}

=item B<get_inlines>

Get all inline expressions referenced in actions of the block.

=cut
sub get_inlines {
    my $self = shift;
    my $action;
    my $inlines;
    
    foreach $action (@{$self->get_actions}) {
      Bi::Utility::push_unique($inlines, $action->get_inlines);
    }
    return $inlines;
}

=item B<get_vars>

Get all variables referenced in actions of the block.

=cut
sub get_vars {
    my $self = shift;
    my $action;
    my $vars = [];
    
    foreach $action (@{$self->get_actions}) {
      Bi::Utility::push_unique($vars, $action->get_target);
      Bi::Utility::push_unique($vars, $action->get_vars);
    }
    return $vars;
}

=item B<push_action>(I<action>)

Add action to block.

=cut
sub push_action {
    my $self = shift;
    my $action = shift;

    assert ($action->isa('Bi::Model::Action')) if DEBUG;
    
    push(@{$self->get_actions}, $action);
}

=item B<push_actions>(I<actions>)

Add actions to block.

=cut
sub push_actions {
    my $self = shift;
    my $args = shift;
    
    assert (ref($args) eq 'ARRAY') if DEBUG;
    
    my $arg;
    foreach $arg (@$args) {
        $self->push_action($arg);
    }
}

=item B<push_block>(I<block>)

Add sub-block to block.

=cut
sub push_block {
    my $self = shift;
    my $block = shift;
    
    assert ($block->isa('Bi::Model::Block')) if DEBUG;
    
    push(@{$self->get_blocks}, $block);
} 

=item B<push_blocks>(I<blocks>)

Add sub-blocks to block.

=cut
sub push_blocks {
    my $self = shift;
    my $args = shift;
    
    assert (ref($args) eq 'ARRAY') if DEBUG;
    
    my $arg;
    foreach $arg (@$args) {
        $self->push_block($arg);
    }
}

=item B<unshift_block>(I<block>)

Add sub-block to start of block.

=cut
sub unshift_block {
    my $self = shift;
    my $block = shift;
    
    assert ($block->isa('Bi::Model::Block')) if DEBUG;
    
    unshift(@{$self->get_blocks}, $block);
} 

=item B<unshift_blocks>(I<blocks>)

Add sub-blocks to start of block.

=cut
sub unshift_blocks {
    my $self = shift;
    my $args = shift;
    
    assert (ref($args) eq 'ARRAY') if DEBUG;
    
    my $arg;
    foreach $arg (@$args) {
        $self->unshift_block($arg);
    }
}

=item B<pop_block>

Remove sub-block from end and return.

=cut
sub pop_block {
    my $self = shift;
    
    return pop(@{$self->get_blocks});
}

=item B<shift_block>

Remove sub-block from start and return.

=cut
sub shift_block {
    my $self = shift;
    
    return shift(@{$self->get_blocks});
}

=item B<can_simulate>

Can block be deterministically simulated?

=cut
sub can_simulate {
    my $self = shift;
    
    return Bi::Visitor::CanSimulate->evaluate($self);
}

=item B<sink_actions>(I<model>)

Sinks all actions into a new sub-block, which is inserted at the end of the
list of blocks. If there are no actions, does nothing.

=cut
sub sink_actions {
    my $self = shift;
    my $model = shift;

    assert (defined($model) && $model->isa('Bi::Model')) if DEBUG;
    
    if ($self->num_actions > 0) {
        my $block = new Bi::Model::Block($model->next_block_id, undef, [], {},
            $self->get_actions);
        $self->set_actions([]);
        $self->push_block($block);
    }
}

=item B<sink_children>(I<model>)

Sinks all sub-blocks and actions into one containing block, which then
becomes the sole child of the block.

=cut
sub sink_children {
    my $self = shift;
    my $model = shift;

    assert (defined($model) && $model->isa('Bi::Model')) if DEBUG;
    
    my $block = new Bi::Model::Block($model->next_block_id, undef, [], {},
        $self->get_actions, $self->get_blocks);
    $self->set_blocks([ $block ]);
    $self->set_actions([]);
}

=item B<add_mean_actions>(I<model>, I<vars>, I<mu>)

Add actions to the block to compute mean terms.

=over 8

=item * I<model>

The model.

=item * I<vars>

Variables included in the mean vector.

=item * I<mu>

The vector of symbolic expressions, as a L<Bi::Expression::Vector> object,
each element either undefined (for zero) or a L<Bi::Expression> object.

=back

No return value.

=cut
sub add_mean_actions {
    my $self = shift;
    my $model = shift;
    my $vars = shift;
    my $mu = shift;
    
    for (my $i = 0; $i < @$vars; ++$i) {
        my $expr = $mu->get($i);
        if (defined $expr) {
            my $id = $model->next_action_id;
            my $var = $vars->[$i];
            my $target = new Bi::Expression::VarIdentifier($var);
            my $action = new Bi::Model::Action($id, $target, '<-', $expr);
            
            $self->push_action($action);
        }
    }
}

=item B<add_std_actions>(I<model>, I<vars>, I<S>)

Add actions to the block to compute square-root covariance terms.

=over 8

=item * I<model>

The model.

=item * I<vars>

Variables included in the matrix.

=item * I<S>

The matrix of symbolic expressions, as a L<Bi::Expression::Matrix> object,
each element either undefined (for zero) or a L<Bi::Expression> object.

=back

No return value.

=cut
sub add_std_actions {
    my $self = shift;
    my $model = shift;
    my $vars = shift;
    my $S = shift;
    
    for (my $i = 0; $i < @$vars; ++$i) {
        for (my $j = 0; $j < @$vars; ++$j) {
            my $expr = $S->get($i, $j);
            if (defined $expr) {
                my $id = $model->next_action_id;
                my $var = $model->get_std_var($vars->[$i], $vars->[$j]);
                my $target = new Bi::Expression::VarIdentifier($var);
                my $action = new Bi::Model::Action($id, $target, '<-', $expr);
                
                $self->push_action($action);
            }
        }
    }
}

=item B<add_jacobian_actions>(I<model>, I<vars>, I<J>)

Add actions to the block to compute Jacobian terms.

=over 8

=item * I<model>

The model.

=item * I<vars>

Variables included in the Jacobian matrix.

=item * I<J>

The matrix of symbolic Jacobian terms, as a L<Bi::Expression::Matrix> object,
each element either undefined (for zero) or a L<Bi::Expression> object.

=back

No return value.

=cut
sub add_jacobian_actions {
    my $self = shift;
    my $model = shift;
    my $vars = shift;
    my $J_commit = shift;
    my $J = shift;
    
    $J = $J_commit*$J;
    
    for (my $i = 0; $i < @$vars; ++$i) {
        for (my $j = 0; $j < @$vars; ++$j) {
            my $expr = $J->get($i, $j);
            if (defined($expr)) {
                my $id = $model->next_action_id;
                my $var = $model->get_jacobian_var($vars->[$i], $vars->[$j]);
                my $target = new Bi::Expression::VarIdentifier($var);
                my $action = new Bi::Model::Action($id, $target, '<-', $expr);
                
                $self->push_action($action);
            }
        }
    }
}

=item B<validate>

Validate arguments.

=cut
sub validate {
    my $self = shift;
    my $name = $self->get_name;
    warn("block '$name' is missing 'validate' method\n");
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    Bi::ArgHandler::accept($self, $visitor, @args);
    @{$self->{_blocks}} = map { $_->accept($visitor, @args) } @{$self->get_blocks};
    @{$self->{_actions}} = map { $_->accept($visitor, @args) } @{$self->get_actions};
    
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
