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
use Bi::Utility qw(find);

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

    $name = 'eval_' unless defined($name); 
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

=item I<i> Action index. Defaults to 0.,

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

=item I<i> Sub-block index. Defaults to 0.,

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

=item B<get_vars>(I<types>)

Get all variables referenced in actions of the block.

=cut
sub get_vars {
    my $self = shift;
    my $types = shift;
    
    my $action;
    my $vars = [];
    
    foreach $action (@{$self->get_actions}) {
      Bi::Utility::push_unique($vars, $action->get_vars($types));
    }
    return $vars;
}

=item B<get_target_vars>(I<types>)

Get all variables targeted in actions of the block.

=cut
sub get_target_vars {
    my $self = shift;
    my $types = shift;
    
    my $action;
    my $vars = [];
    
    foreach $action (@{$self->get_actions}) {
      Bi::Utility::push_unique($vars, $action->get_left->get_vars($types));
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

sub add_extended_actions {
	my $self = shift;	
	my $model = shift;
    my $vars = shift;
    my $J_commit = shift;
    my $J_new = shift;
    
    my $N = $J_commit->num_cols;
    my $mu = new Bi::Expression::Vector($N);
    my $S = new Bi::Expression::Matrix($N, $N);
    my $J = new Bi::Expression::Matrix($N, $N);

    # get and then clear actions, will be replaced
    my $actions = $self->get_actions;
    $self->set_actions([]);
    
    foreach my $action (@$actions) {
        # search for index that corresponds to the target of this action
        my $j = find($vars, $action->get_left->get_var);
        assert ($j >= 0);

        $J_new->set($j, $j, new Bi::Expression::Literal(0));
        $J->set($j, $j, new Bi::Expression::Literal(0));            
        
        # mean
        my $mean = $action->mean;
        if (defined $mean) {
            $mu->set($j, $mean);

            my $id = $model->next_action_id;
            my $target = $action->get_target->clone;
            my $action = new Bi::Model::Action($id, $target, '<-', $mean);
            
            $self->push_action($action);
        }

        # square-root covariance
        my $std = $action->std;
        if (defined $std) {
            $S->set($j, $j, $std);
            $J_new->set($j, $j, new Bi::Expression::Literal(1));
            $J->set($j, $j, undef);
            
            my $id = $model->next_action_id;
            my $var = $model->get_std_var($vars->[$j], $vars->[$j]);
            my @aliases = map { $_->clone } (@{$action->get_target->get_aliases}, @{$action->get_target->get_aliases});
            my $target = new Bi::Model::Target($var, \@aliases);
            my $action = new Bi::Model::Action($id, $target, '<-', $std);
                
            $self->push_action($action);
        }

        # Jacobian
        my ($ds, $refs) = $action->jacobian;

        for (my $l = 0; $l < $J_commit->num_rows; ++$l) {
        	my $expr;
	        for (my $k = 0; $k < @$ds; ++$k) {
	            my $ref = $refs->[$k];
	            my $d = $ds->[$k];
	            my $i = find($vars, $ref->get_var);
	            
	            my $inline = $model->lookup_inline($d);
                if (!defined $inline) {
                    $inline = Bi::Model::Inline->new($model->tmp_inline, $d);
                    $model->add_inline($inline);
                }
                $d = new Bi::Expression::InlineIdentifier($inline);
	            
	            if ($i >= 0) {
	                $J_new->set($i, $j, $d->clone);
                
                	if (defined $J_commit->get($l, $i)) {
                        my $arg = $J_commit->get($l, $i);
                        my $indexes1 = [];
                        my $indexes2 =  $ref->get_indexes;
                        if (@$indexes2) {
                           $indexes1 = $vars->[$l]->gen_indexes;
                	    }
		                my @indexes = map { $_->clone } (@{$indexes1}, @{$indexes2});
		                $arg->set_indexes(\@indexes);

						if (!defined $expr) {
							$expr = $d->clone*$arg;
						} else {
	                        $expr += $d->clone*$arg;
						}
                	}
                }
            }
            if (defined $expr) {
            	my $id = $model->next_action_id;
		        my $var = $model->get_jacobian_var($vars->[$l], $vars->[$j]);
		        my $aliases1 = [];
		        my $aliases2 = $action->get_target->get_aliases;
		        if (@$aliases2) {
		            $aliases1 = $vars->[$l]->gen_aliases;
		        }
		        my @aliases = map { $_->clone } (@$aliases1, @$aliases2);
		        my $target = new Bi::Model::Target($var, \@aliases);
	            my $action = new Bi::Model::Action($id, $target, '<-', $expr);
	            $self->push_action($action);
            }
        }
    }
    #_inline($model, $J);
    #$self->add_mean_actions($model, $vars, $mu);
    #$self->add_std_actions($model, $vars, $S);
    #$self->add_jacobian_actions($model, $vars, $J_commit, $J);

    if ($self->get_commit) {
        $J_commit->swap(Bi::Jacobian::commit($model, $vars, $J_commit*$J_new));
        $J_new->ident;
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

=item B<equals>(I<obj>)

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return ref($obj) eq ref($self) && $self->get_id eq $obj->get_id;
}

1;

=back

=head1 SEE ALSO

L<Bi::Model>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
