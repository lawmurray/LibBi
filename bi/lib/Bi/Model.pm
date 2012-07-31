=head1 NAME

Bi::Model - model specification

=head1 SYNOPSIS

    use Bi::Model;

=head1 INHERITS

L<Bi::ArgHandler>

=head1 METHODS

=over 4

=cut

package Bi::Model;

use base 'Bi::ArgHandler';
use warnings;
use strict;

use Carp::Assert;
use Bi::Model::Action;
use Bi::Model::Block;
use Bi::Model::Const;
use Bi::Model::Inline;
use Bi::Model::Dim;
use Bi::Model::State;
use Bi::Model::StateAux;
use Bi::Model::Noise;
use Bi::Model::Input;
use Bi::Model::Obs;
use Bi::Model::Param;
use Bi::Model::ParamAux;
use Bi::Model::Spec;
use Bi::Model::Var;
use Bi::Expression;

our %MAP_BLOCKS = (
	'lookahead_transition' => 'transition',
	'lookahead_observation' => 'observation',
	'proposal_parameter' => 'proposal',
	'proposal_initial' => 'initial'
);

=item B<new>

Empty constructor.

=cut
sub new {
    my $class = shift;
    my $self = new Bi::ArgHandler();
    
    $self->{_name} = '';
    $self->{_blocks} = [];
    $self->{_named_blocks} = {};

    $self->{_dims} = [];
    $self->{_dim_names} = {};

    $self->{_consts} = [];
    $self->{_const_names} = {};
        
    $self->{_inlines} = [];
    $self->{_inline_names} = {};

    $self->{_vars} = [];
    $self->{_state_vars} = [];
    $self->{_state_aux__vars} = [];
    $self->{_noise_vars} = [];
    $self->{_input_vars} = [];
    $self->{_obs_vars} = [];
    $self->{_param_vars} = [];
    $self->{_param_aux__vars} = [];
        
    $self->{_var_names} = {};
    $self->{_state_names} = {};
    $self->{_state_aux__names} = {};
    $self->{_noise_names} = {};
    $self->{_input_names} = {};
    $self->{_obs_names} = {};
    $self->{_param_names} = {};
    $self->{_param_aux__names} = {};
        
    $self->{_block_id} = 0;
    $self->{_action_id} = 0;
    $self->{_tmp_id} = 0;

    bless $self, $class;
    
    $self->validate;
    
    return $self;
}

=item B<init>(I<name>, I<args>, I<named_args>, I<blocks>, I<consts>,
I<inlines>)

Initialise model after construction.

=cut
sub init {
    my $self = shift;
    my $name = shift;
    my $args = shift;
    my $named_args = shift;
    my $blocks = shift;
    my $consts = shift;
    my $inlines = shift;
    
    # pre-condition
    assert(!defined($args) || ref($args) eq 'ARRAY') if DEBUG;
    assert(!defined($named_args) || ref($named_args) eq 'HASH') if DEBUG;
    map { assert($_->isa('Bi::Model::Block')) if DEBUG } @$blocks;
    map { assert($_->isa('Bi::Model::Const')) if DEBUG } @$consts;
    map { assert($_->isa('Bi::Model::Inline')) if DEBUG } @$inlines;
        
    $self->{_name} = $name;
    $self->{_args} = $args;
    $self->{_named_args} = $named_args;
    
    $self->_rebuild_blocks($blocks);
    $self->_rebuild_consts($consts);
    $self->_rebuild_inlines($inlines);
    
    # complete top-level blocks
    foreach my $name (keys %MAP_BLOCKS) {
    	if (!$self->is_block($name)) {
    		my $block = $self->get_block($MAP_BLOCKS{$name})->clone($self);
    		$block->set_name($name);
    		$self->add_block($block);
    	}
    }
}

=item B<get_name>

Get the name of the model.

=cut
sub get_name {
    my $self = shift;
    return $self->{_name};
}

=item B<get_dim>(I<name>)

Get dimension of the given I<name> as L<Bi::Model::Dim> object.

=cut
sub get_dim {
    my $self = shift;
    my $name = shift;
    
    assert ($self->is_dim($name)) if DEBUG;
    
    return $self->{_dim_names}->{$name};
}

=item B<get_const>(I<name>)

Get constant of the given I<name> as L<Bi::Model::Const> object.

=cut
sub get_const {
    my $self = shift;
    my $name = shift;
    
    assert ($self->is_const($name)) if DEBUG;
    
    return $self->{_const_names}->{$name};
}

=item B<get_inline>(I<name>)

Get inline expression of the given I<name> as L<Bi::Model::Inline> object.

=cut
sub get_inline {
    my $self = shift;
    my $name = shift;
    
    assert ($self->is_inline($name)) if DEBUG;

    return $self->{_inline_names}->{$name};
}

=item B<get_var>(I<name>)

Get variable of the given I<name> as L<Bi::Model::Var> object.

=cut
sub get_var {
    my $self = shift;
    my $name = shift;
    
    assert ($self->is_var($name)) if DEBUG;
    
    return $self->{_var_names}->{$name};
}

=item B<get_block>(I<name>)

Get block of the given I<name> as L<Bi::Model::Block> object.

=cut
sub get_block {
    my $self = shift;
    my $name = shift;
    
    assert ($self->is_block($name)) if DEBUG;
    
    return $self->{_named_blocks}->{$name};
}

=item B<get_dims>

Get ordered list of the dimensions of the model as L<Bi::Model::Dim>
objects.

=cut
sub get_dims {
    my $self = shift;
    return $self->{_dims};
}

=item B<get_consts>

Get ordered list of the constants of the model as L<Bi::Model::Const>
objects.

=cut
sub get_consts {
    my $self = shift;
    return $self->{_consts};
}

=item B<get_inlines>

Get ordered list of the inlines of the model as L<Bi::Model::Inline>
objects.

=cut
sub get_inlines {
    my $self = shift;
    return $self->{_inlines};
}

=item B<get_vars>(I<types>)

Get ordered list of the variables of the model as L<Bi::Model::Variable>
objects. If I<type> is given as a string, only variables of that type are
returned. If I<type> is given as an array ref of strings, only variables of
those types are returned.

=cut
sub get_vars {
    my $self = shift;
    my $types = [];
    if (@_) {
      $types = shift;
    }
    if (ref($types) ne 'ARRAY') {
        $types = [ $types ];
    }
    
    my $vars = [];
    my $type;
    
    if (@$types) {
        foreach $type (@$types) {
            assert (exists $self->{"_${type}_vars"}) if DEBUG;
            push(@$vars, @{$self->{"_${type}_vars"}});
        }
    } else {
        $vars = $self->{_vars};
    }
    
    return $vars;
}

=item B<get_args>

Get ordered list of positional arguments as L<Bi::Expression> objects.

=cut
sub get_args {
    my $self = shift;
    return $self->{_args};
}

=item B<get_blocks>

Get ordered list of top-level blocks as I<Bi::Model::Block> objects.

=cut
sub get_blocks {
    my $self = shift;
    return $self->{_blocks};
}

=item B<get_size>(I<type>)

Get the size of the model. If I<type> is given, the size of the net for that
particular type is returned, otherwise the sum of all sizes is returned.

=cut
sub get_size {
    my $self = shift;
    my $type;
    if (@_) {
      $type = shift;
    }

    my $vars = $self->get_vars($type);
    my $var;
    my $size = 0;
    foreach $var (@$vars) {
        $size += $var->get_size;
    }

    return $size;
}

=item B<add_dim>(I<dim>)

Add dimension.

=cut
sub add_dim {
    my $self = shift;
    my $dim = shift;
    my $name = $dim->get_name;
              
    # pre-condition
    assert($dim->isa('Bi::Model::Dim')) if DEBUG;
    
    if ($self->is_dim($name)) {
        die("dimension '$name' already declared");
    } else {
        $dim->set_id(scalar(@{$self->{_dims}}));
        $self->{_dim_names}->{$name} = $dim;
        push(@{$self->{_dims}}, $dim);
    }
}

=item B<add_const>(I<const>)

Add constant.

=cut
sub add_const {
    my $self = shift;
    my $const = shift;
    my $name = $const->get_name;
    
    # pre-condition
    assert($const->isa('Bi::Model::Const')) if DEBUG;

    if ($self->is_var($name)) {
        die("variable '$name' already declared");
    } elsif ($self->is_const($name)) {
        die("constant '$name' already declared");
    } elsif ($self->is_inline($name)) {
        die("inline expression '$name' already declared");
    } else {
        $self->{_const_names}->{$name} = $const;
        push(@{$self->{_consts}}, $const);
    }
}

=item B<add_inline>(I<inline>)

Add inline.

=cut
sub add_inline {
    my $self = shift;
    my $inline = shift;
    my $name = $inline->get_name;
    
    # pre-condition
    assert($inline->isa('Bi::Model::Inline')) if DEBUG;

    if ($self->is_var($name)) {
        die("variable '$name' already declared");
    } elsif ($self->is_const($name)) {
        die("constant '$name' already declared");
    } elsif ($self->is_inline($name)) {
        die("inline expression '$name' already declared");
    } else {
        $self->{_inline_names}->{$name} = $inline;
        push(@{$self->{_inlines}}, $inline);
    }
}

=item B<add_var>(I<var>)

Add variable.

=cut
sub add_var {
    my $self = shift;
    my $var = shift;
    my $name = $var->get_name;
    
    if ($self->is_var($name)) {
        die("variable '$name' already declared");
    } elsif ($self->is_const($name)) {
        die("constant '$name' already declared");
    } elsif ($self->is_inline($name)) {
        die("inline expression '$name' already declared");
    } else {
        my $type = $var->get_type;
        assert (exists $self->{"_${type}_names"}) if DEBUG;
        
          $var->set_id(scalar(@{$self->{"_${type}_vars"}}));
        push(@{$self->{"_${type}_vars"}}, $var);
        $self->{"_${type}_names"}->{$name} = $var;
        push(@{$self->{_vars}}, $var);
        $self->{_var_names}->{$name} = $var;
    }
}

=item B<add_block>(I<block>)

Add block.

=cut
sub add_block {
    my $self = shift;
    my $block = shift;
    my $name = $block->get_name;
              
    # pre-condition
    assert($block->isa('Bi::Model::Block')) if DEBUG;
    
    if ($self->is_block($name)) {
        die("block '$name' already declared");
    } else {
        $self->{_named_blocks}->{$name} = $block;
        push(@{$self->{_blocks}}, $block);
    }
}

=item B<is_dim>(I<name>)

Is there a dimension of the given name?

=cut
sub is_dim {
    my $self = shift;
    my $name = shift;
    
    return exists $self->{_dim_names}->{$name};
}

=item B<is_const>(I<name>)

Is there a constant of the given name?

=cut
sub is_const {
    my $self = shift;
    my $name = shift;
    
    return exists $self->{_const_names}->{$name};
}

=item B<is_inline>(I<name>)

Is there an inline expression of the given name?

=cut
sub is_inline {
    my $self = shift;
    my $name = shift;
    
    return exists $self->{_inline_names}->{$name};
}

=item B<is_var>(I<name>)

Is there a variable (of any type) of the given name?

=cut
sub is_var {
    my $self = shift;
    my $name = shift;
    
    return exists $self->{_var_names}->{$name};
}

=item B<is_state>(I<name>)

Is there a state variable of the given name?

=cut
sub is_state {
    my $self = shift;
    my $name = shift;
    
    return exists $self->{_state_names}->{$name};
}

=item B<is_state_aux>(I<name>)

Is there an auxiliary state variable of the given name?

=cut
sub is_state_aux {
    my $self = shift;
    my $name = shift;
    
    return exists $self->{_state_aux_names}->{$name};
}

=item B<is_noise>(I<name>)

Is there a noise variable of the given name?

=cut
sub is_noise {
    my $self = shift;
    my $name = shift;
    
    return exists $self->{_noise_names}->{$name};
}

=item B<is_input>(I<name>)

Is there an input of the given name?

=cut
sub is_input {
    my $self = shift;
    my $name = shift;
    
    return exists $self->{_input_names}->{$name};
}

=item B<is_obs>(I<name>)

Is there an observation of the given name?

=cut
sub is_obs {
    my $self = shift;
    my $name = shift;
    
    return exists $self->{_obs_names}->{$name};
}

=item B<is_param>(I<name>)

Is there a parameter of the given name?

=cut
sub is_param {
    my $self = shift;
    my $name = shift;
    
    return exists $self->{_param_names}->{$name};
}

=item B<is_param_aux>(I<name>)

Is there an auxiliary parameter of the given name?

=cut
sub is_param_aux {
    my $self = shift;
    my $name = shift;
    
    return exists $self->{_param_aux_names}->{$name};
}

=item B<is_block>(I<name>)

Is there a block of the given name?

=cut
sub is_block {
    my $self = shift;
    my $name = shift;
        
    return exists $self->{_named_blocks}->{$name};
}

=item B<get_var_start>(I<var>)

Get the starting index of variable I<var> among those of its type. This is
the sum of the sizes of all preceding variables.

=cut
sub get_var_start {
    my $self = shift;
    my $var = shift;
    
    assert ($self->is_var($var->get_name)) if DEBUG;
    
    my $type = $var->get_type;
    my $var2;
    my $start = 0;
    foreach $var2 (@{$self->get_vars($type)}) {
        if ($var->equals($var2)) {
            last;
        } else {
            $start += $var2->get_size;
        }
    }
    
    return $start;
}

=item B<num_dims>

Number of dimensions.

=cut
sub num_dims {
    my $self = shift;
    
    return scalar(@{$self->{_dims}});
}

=item B<num_consts>

Number of constants.

=cut
sub num_consts {
    my $self = shift;
    
    return scalar(@{$self->{_consts}});
}

=item B<num_inlines>

Number of inlines.

=cut
sub num_inlines {
    my $self = shift;
    
    return scalar(@{$self->{_inlines}});
}

=item B<num_vars>

Number of vars.

=cut
sub num_vars {
    my $self = shift;
    
    return scalar(keys %{$self->{_var_names}});
}

=item B<next_block_id>

Get next block id and increment counter.

=cut
sub next_block_id {
    my $self = shift;
    
    return $self->{_block_id}++;
}

=item B<next_action_id>

Get next action id and increment counter.

=cut
sub next_action_id {
    my $self = shift;
    
    return $self->{_action_id}++;
}

=item B<tmp_dim>

Generate an arbitrary name for an anonymous dimension.

=cut
sub tmp_dim {
    my $self = shift;
    my $name;
    
    do {
        $name = 'dim_' . sprintf("%x", $self->{_tmp_id}++) . '_';
    } until (!exists $self->{_dim_names}->{$name});
    
    return $name;
}

=item B<tmp_var>

Generate an arbitrary name for an anonymous variable.

=cut
sub tmp_var {
    my $self = shift;
    my $name;
    
    do {
        $name = 'var_' . sprintf("%x", $self->{_tmp_id}++) . '_';
    } until (!exists $self->{_var_names}->{$name});
    
    return $name;
}

=item B<validate>

Validate model.

=cut
sub validate {
    my $self = shift;
    
    #
}

=item B<accept>(I<visitor>, ...)

Accept visitor.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    Bi::ArgHandler::accept($self, $visitor, @args);

    my @dims = map { $_->accept($visitor, @args) } @{$self->get_dims};
    $self->_rebuild_dims(\@dims);

    my @consts = map { $_->accept($visitor, @args) } @{$self->get_consts};
    $self->_rebuild_consts(\@consts);

    my @vars = map { $_->accept($visitor, @args) } @{$self->get_vars};
    $self->_rebuild_vars(\@vars);

    my @inlines = map { $_->accept($visitor, @args) } @{$self->get_inlines};
    $self->_rebuild_inlines(\@inlines);

    my @blocks = map { $_->accept($visitor, @args) } @{$self->get_blocks};    
    $self->_rebuild_blocks(\@blocks);

    return $visitor->visit($self, @args);
}

=item B<_rebuild_dims>(I<dims>)

Rebuild dimension attributes.

=cut
sub _rebuild_dims {
    my $self = shift;
    my $dims = shift;

    $self->{_dims} = [];
    $self->{_dim_names} = {};

    foreach my $dim (@$dims) {
        $self->add_dim($dim);
    }
}

=item B<_rebuild_consts>(I<consts>)

Rebuild constant attributes.

=cut
sub _rebuild_consts {
    my $self = shift;
    my $consts = shift;

    $self->{_consts} = [];
    $self->{_const_names} = {};

    foreach my $const (@$consts) {
        $self->add_const($const);
    }
}

=item B<_rebuild_vars>(I<vars>)

Rebuild variable attributes.

=cut
sub _rebuild_vars {
    my $self = shift;
    my $vars = shift;

    $self->{_vars} = [];
    $self->{_state_vars} = [];
    $self->{_state_aux__vars} = [];
    $self->{_noise_vars} = [];
    $self->{_input_vars} = [];
    $self->{_obs_vars} = [];
    $self->{_param_vars} = [];
    $self->{_param_aux__vars} = [];
        
    $self->{_var_names} = {};
    $self->{_state_names} = {};
    $self->{_state_aux__names} = {};
    $self->{_noise_names} = {};
    $self->{_input_names} = {};
    $self->{_obs_names} = {};
    $self->{_param_names} = {};
    $self->{_param_aux__names} = {};

    foreach my $var (@$vars) {
        $self->add_var($var);
    }
}

=item B<_rebuild_inlines>(I<inlines>)

Rebuild inline attributes.

=cut
sub _rebuild_inlines {
    my $self = shift;
    my $inlines = shift;

    $self->{_inlines} = [];
    $self->{_inline_names} = {};

    foreach my $inline (@$inlines) {
        $self->add_inline($inline);
    }
}

=item B<_rebuild_blocks>(I<blocks>)

Rebuild block attributes.

=cut
sub _rebuild_blocks {
    my $self = shift;
    my $blocks = shift;

    $self->{_blocks} = [];
    $self->{_named_blocks} = {};

    foreach my $block (@$blocks) {
        $self->add_block($block);
    }
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
