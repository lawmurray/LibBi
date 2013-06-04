=head1 NAME

Bi::Block - block.

=head1 SYNOPSIS

    use Bi::Block;
    
    my $block = Bi::Block;
    $block->set_name(...);
    $block->push_children([ ... ]);
    $block->validate;

=head1 INHERITS

L<Bi::ArgHandler>

=head1 METHODS

=over 4

=cut

package Bi::Block;

use parent 'Bi::Node', 'Bi::ArgHandler';
use warnings;
use strict;

use Carp::Assert;
use Bi::Action;
use Bi::Utility qw(find);

use Bi::Model::Dim;
use Bi::Model::Var;
use Bi::Model::Const;
use Bi::Model::Inline;

our $_next_block_id = 0;

=item B<new>

Constructor.

=cut
sub new {
    my $class = shift;

    my $self = new Bi::ArgHandler;
    $self->{_id} = $_next_block_id++;
    $self->{_name} = undef;
    $self->{_toplevel} = 0;
    $self->{_consts} = [];
    $self->{_inlines} = [];
    $self->{_dims} = [];
    $self->{_vars} = [];
    $self->{_children} = [];

    bless $self, $class;    
    return $self;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;

    my $clone = Bi::ArgHandler::clone($self);
    $clone->{_id} = $_next_block_id++;
    $clone->{_name} = $self->get_name;
    $clone->{_toplevel} = $self->get_top_level;
    $clone->{_consts} = $self->get_consts;
    $clone->{_inlines} = $self->get_inlines;
    $clone->{_dims} = $self->get_dims;
    $clone->{_vars} = $self->get_vars;
    $clone->{_children} = [ map { $_->clone } @{$self->get_children} ];
    
    bless $clone, ref($self);
    return $clone; 
}

=item B<clear>

Clear contents of block.

=cut
sub clear {
    my $self = shift;
    
    $self->{_consts} = [];
    $self->{_inlines} = [];
    $self->{_dims} = [];
    $self->{_vars} = [];
    $self->{_children} = [];
}

=item B<get_id>

Get the id of the block.

=cut
sub get_id {
    my $self = shift;
    return $self->{_id};
}

=item B<get_type>

Returns the string "block".

=cut
sub get_type {
    return "block";
}

=item B<get_name>

Get the name of the block.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<is_named>

Is this a named block?

=cut
sub is_named {
	my $self = shift;
	return defined($self->get_name);
}

=item B<set_name>(I<name>)

Set the name of the block.

A change in name will morph the object into a new type derived from
L<Bi::Block>.

=cut
sub set_name {
    my $self = shift;
    my $name = shift;
    
    # morph to appropriate class for name
	my $class = 'Bi::Block';
	if (defined $name) {
	    if ($name !~ /^\w+$/) {
	    	die("don't know what to do with block '$name'\n");
	    } else {
	    	$class = "Bi::Block::$name";
	        eval("require $class") || die("don't know what to do with block '$name'\n");
	    }
	}
    bless $self, $class;

    $self->{_name} = $name;
}

=item B<get_top_level>

Is this a top-level block?

=cut
sub get_top_level {
	my $self = shift;
	
	return $self->{_toplevel};
}

=item B<set_top_level>(I<flag>)

Set whether this is a top-level block.

=cut
sub set_top_level {
	my $self = shift;
	my $flag = shift;
	
	$self->{_toplevel} = $flag;
}

=back

=head2 Constants

=over 4

=item B<get_consts>

Get all constants declared in the block.

=cut
sub get_consts {
    my $self = shift;
    
    return $self->{_consts};
}

=item B<get_const>(I<name>)

Get the constant called I<name>, or undef if it does not exist.

=cut
sub get_const {
    my $self = shift;
    my $name = shift;

    return $self->_get_item($self->get_consts, $name);
}

=item B<is_const>(I<name>)

Is there a constant called I<name>?

=cut
sub is_const {
    my $self = shift;
    my $name = shift;
    
    return $self->_is_item($self->get_consts, $name);
}

=item B<push_const>(I<const>)

Add a constant.

=cut
sub push_const {
    my $self = shift;
    my $const = shift;
    
    push(@{$self->get_consts}, $const);
}

=back

=head2 Inlines

=over 4

=item B<get_inlines>

Get all inline expressions declared in the block.

=cut
sub get_inlines {
    my $self = shift;
    
    return $self->{_inlines};
}

=item B<get_inline>(I<name>)

Get the inline expression called I<name>, or undef if it does not exist.

=cut
sub get_inline {
    my $self = shift;
    my $name = shift;

    return $self->_get_item($self->get_inlines, $name);
}

=item B<is_inline>(I<name>)

Is there an inline expression called I<name>?

=cut
sub is_inline {
    my $self = shift;
    my $name = shift;
    
    return $self->_is_item($self->get_inlines, $name);
}

=item B<push_inline>(I<inline>)

Add an inline expression.

=cut
sub push_inline {
    my $self = shift;
    my $inline = shift;
    
    push(@{$self->get_inlines}, $inline);
}

=item B<lookup_inline>(I<expr>)

Search for an inline expression that has the same expression as I<expr>,
and return it. If such an inline expression does not exist, create one and
return it.

=cut
sub lookup_inline {
    my $self = shift;
    my $expr = shift;
    
    assert ($expr->isa('Bi::Expression')) if DEBUG;

    my $inline;
    foreach $inline (@{$self->get_all_inlines}) {
        if ($inline->get_expr->equals($expr)) {
            return $inline;
        }
    }
    $inline = new Bi::Model::Inline(undef, $expr);
    $self->push_inline($inline);
    
    return $inline;
}

=back

=head2 Dimensions

=over 4

=item B<get_dims>

Get all dimensions declared in the block.

=cut
sub get_dims {
    my $self = shift;
    
    return $self->{_dims};
}

=item B<get_dim>(I<dim>)

Get the dimension called I<name>, or undef if it does not exist.

=cut
sub get_dim {
    my $self = shift;
    my $name = shift;

    return $self->_get_item($self->get_dims, $name);
}

=item B<is_dim>(I<name>)

Is there a dimension called I<name>?

=cut
sub is_dim {
    my $self = shift;
    my $name = shift;
    
    return $self->_is_item($self->get_dims, $name);
}

=item B<push_dim>(I<dim>)

Add a dimension.

=cut
sub push_dim {
    my $self = shift;
    my $dim = shift;
    
    push(@{$self->get_dims}, $dim);
}

=item B<lookup_dim>(I<size>)

Search for a dimension of size I<size>, with an ordinary boundary condition,
and return it. If such a dimension does not exist, create one and return it.

=cut
sub lookup_dim {
    my $self = shift;
    my $size = shift;
    
    my $dim;
    foreach $dim (@{$self->get_all_dims}) {
        if ($dim->get_size == $size &&
                $dim->get_named_arg('boundary')->eval_const eq 'none') {
            return $dim;
        }
    }
    $dim = new Bi::Model::Dim(undef, [ new Bi::Expression::IntegerLiteral($size) ], {});
    $self->push_dim($dim);
    
    return $dim;
}

=back

=head2 Variables

=over 4

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
    
    push(@{$self->get_vars}, $var);
}

=back

=head2 Children (actions and blocks)

=over 4

=item B<get_children>

Get all children declared in the block.

=cut
sub get_children {
    my $self = shift;
    return $self->{_children};
}

=item B<set_children>(I<children>)

Set all children.

=cut
sub set_children {
    my $self = shift;
    my $children = shift;
    $self->{_children} = $children;
}

=item B<push_child>(I<child>)

Add child.

=cut
sub push_child {
    my $self = shift;
    my $child = shift;

    assert ($child->isa('Bi::Action') || $child->isa('Bi::Block')) if DEBUG;
    
    push(@{$self->get_children}, $child);
}

=item B<push_children>(I<children>)

Add children.

=cut
sub push_children {
    my $self = shift;
    my $children = shift;
    
    assert (ref($children) eq 'ARRAY') if DEBUG;
    map { assert ($_->isa('Bi::Action') || $_->isa('Bi::Block')) } @$children if DEBUG;    

    push(@{$self->get_children}, @$children);
}

=item B<unshift_child>(I<child>)

Add child.

=cut
sub unshift_child {
    my $self = shift;
    my $child = shift;

    assert ($child->isa('Bi::Action') || $child->isa('Bi::Block')) if DEBUG;
    
    unshift(@{$self->get_children}, $child);
}

=item B<unshift_children>(I<children>)

Add children.

=cut
sub unshift_children {
    my $self = shift;
    my $children = shift;
    
    assert (ref($children) eq 'ARRAY') if DEBUG;
    map { assert ($_->isa('Bi::Action') || $_->isa('Bi::Block')) } @$children if DEBUG;
    
    unshift(@{$self->get_children}, @$children);
}

=item B<shift_child>

Remove child.

=cut
sub shift_child {
    my $self = shift;

    return shift(@{$self->get_children});
}

=item B<pop_child>

Remove child.

=cut
sub pop_child {
    my $self = shift;

    return pop(@{$self->get_children});
}

=item B<get_actions>

Get actions.

=cut
sub get_actions {
    my $self = shift;
    
    return [ map { ($_->isa('Bi::Action')) ? $_ : () } @{$self->get_children} ];
}

=item B<get_blocks>

Get blocks.

=cut
sub get_blocks {
    my $self = shift;
    
    return [ map { ($_->isa('Bi::Block')) ? $_ : () } @{$self->get_children} ];
}

=item B<get_block>(I<name>)

Get the first block called I<name>, or undef if no such block exists.

=cut
sub get_block {
    my $self = shift;
    my $name = shift;

    return $self->_get_item($self->get_blocks, $name);
}

=item B<is_block>(I<name>)

Is there a block called I<name>?

=cut
sub is_block {
    my $self = shift;
    my $name = shift;
    
    return $self->_is_item($self->get_blocks, $name);
}

=item B<validate>

Validate arguments.

=cut
sub validate {
    my $self = shift;
    my $name = $self->get_name;
    
    if (defined $name) {
	    warn("block '$name' is missing 'validate' method\n");
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

    @{$self->{_consts}} = map { $_->accept($visitor, @args) } @{$self->get_consts};
    @{$self->{_inlines}} = map { $_->accept($visitor, @args) } @{$self->get_inlines};
    @{$self->{_dims}} = map { $_->accept($visitor, @args) } @{$self->get_dims};
    @{$self->{_vars}} = map { $_->accept($visitor, @args) } @{$self->get_vars};
    
    # visiting each child may return zero or more children, map works here
    @{$self->{_children}} = map { $_->accept($visitor, @args) } @{$self->get_children};

    return $visitor->visit_after($self, @args);
}

=item B<equals>(I<obj>)

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    return ref($obj) eq ref($self) && $self->get_id == $obj->get_id;
}

=item B<_is_item>(I<list>, I<name>)

Is there an item called I<name> in I<list>?

=cut
sub _is_item {
    my $self = shift;
    my $list = shift;
    my $name = shift;
    
    my $result = 0;
    foreach my $item (@$list) {
        $result = $result || $item->get_name eq $name;
    }
    return $result;
}

=item B<_get_item>(I<list>, I<name>)

Get the item called I<name> from I<list>, or undef if it does not exist.

=cut
sub _get_item {
    my $self = shift;
    my $list = shift;
    my $name = shift;
    
    foreach my $item (@$list) {
        if ($item->get_name eq $name) {
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
