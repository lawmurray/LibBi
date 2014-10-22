=head1 NAME

Bi::Gen::Cpp - generator for C++ source files.

=head1 SYNOPSIS

    use Bi::Gen::Cpp;
    $gen = new Bi::Gen::Cpp($outdir);
    $gen->gen($model);

=head1 INHERITS

L<Bi::Gen>

=head1 METHODS

=over 4

=cut

package Bi::Gen::Cpp;

use parent 'Bi::Gen';
use warnings;
use strict;

use Carp::Assert;
use File::Spec;
use POSIX qw(ceil);

use Bi qw(share_file share_dir);
use Bi::Gen;
use Bi::Model;
use Bi::Expression;
use Bi::Visitor::ToCpp;
use Bi::Visitor::ToAscii;
use Bi::Visitor::ToSymbolic;

=item B<new>(I<outdir>)

Constructor.

=cut
sub new {
    my $class = shift;
    my $outdir = shift;

    my $ttdir = share_dir(File::Spec->catdir('tt', 'cpp'));
    my $self = Bi::Gen->new($ttdir, $outdir);
    $self->get_tt->context->define_vmethod('list', 'to_cpp', \&to_cpp);
    $self->get_tt->context->define_vmethod('hash', 'to_cpp', \&to_cpp);
    $self->get_tt->context->define_vmethod('list', 'to_ascii', \&to_ascii);
    $self->get_tt->context->define_vmethod('hash', 'to_ascii', \&to_ascii);
    $self->get_tt->context->define_vmethod('array', 'to_typetree', \&to_typetree);
    $self->get_tt->context->define_filter('to_camel_case', \&to_camel_case);
    
    bless $self, $class;
    return $self;
}

=item B<gen>(I<model>, I<client>)

Generate code for model and client.

=cut
sub gen {
    my $self = shift;
    my $model = shift;
    my $client = shift;
    my $out;
    
    # pre-condition
    assert(!defined $model || $model->isa('Bi::Model')) if DEBUG;
    assert(!defined $client || $client->isa('Bi::Client')) if DEBUG;

    # model
    if (defined $model) {
        $out = File::Spec->catfile('src', 'model', 'Model' . $model->get_name);
        $self->process_templates('model', { 'model' => $model }, $out);

        # dimensions
        foreach my $dim (@{$model->get_all_dims}) {
            $self->process_dim($model, $dim);
        }
        
        # variables
        foreach my $var (@{$model->get_all_vars}) {
            $self->process_var($model, $var);
        }

        # variable groups
        foreach my $group (@{$model->get_all_var_groups}) {
            $self->process_var_group($model, $group);
        }
    
        # blocks
        foreach my $block (@{$model->get_all_blocks}) {
            if ($block != $model) {
                $self->process_block($model, $block);
            }
        }

        # actions
        foreach my $action (@{$model->get_all_actions}) {
            $self->process_action($model, $action);
        }
    }
        
    # client
    if (defined $client) {
        $self->process_client($model, $client);
    }
    
    # library
    $self->copy_dir('src', 'src', ['cpp', 'hpp', 'cu', 'cuh']);
}

=item B<process_dim>(I<dim>)

Generate code for dimension.

=cut
sub process_dim {
    my $self = shift;
    my $model = shift;
    my $dim = shift;

    my $template = 'dim';
    my $out = File::Spec->catfile('src', 'model', 'dim', 'Dim' . $dim->get_id);
    $self->process_templates($template, { 'dim' => $dim, 'model' => $model }, $out);   
}

=item B<process_var>(I<var>)

Generate code for variable.

=cut
sub process_var {
    my $self = shift;
    my $model = shift;
    my $var = shift;

    my $template = 'var';
    my $out = File::Spec->catfile('src', 'model', 'var', 'Var' . $var->get_id);
    $self->process_templates($template, { 'var' => $var, 'model' => $model }, $out);    

    $template = 'var_coord';
    $out = File::Spec->catfile('src', 'model', 'var', 'VarCoord' . $var->get_id);
    $self->process_templates($template, { 'var' => $var, 'model' => $model }, $out);    
}

=item B<process_var_group>(I<group>)

Generate code for variable group.

=cut
sub process_var_group {
    my $self = shift;
    my $model = shift;
    my $group = shift;

    my $template = 'var_group';
    my $out = File::Spec->catfile('src', 'model', 'var', 'VarGroup' . $group->get_name);
    $self->process_templates($template, { 'var_group' => $group, 'model' => $model }, $out);
}

=item B<process_block>(I<block>)

Generate code for block.

=cut
sub process_block {
    my $self = shift;
    my $model = shift;
    my $block = shift;

    my $template;
    my $out;
    my $child;

    if (defined($block->get_name)) {
        $template = File::Spec->catfile('block', lc($block->get_name));
        if (!$self->is_template("$template.hpp.tt")) {
            die("don't know what to do with block '" . $block->get_name . "'\n");
        }
    } else {
        die("unnamed block\n");
    }
    
    $out = File::Spec->catfile('src', 'model', 'block', 'Block' . $block->get_id);
    $self->process_templates($template, { 'block' => $block, 'model' => $model }, $out);
}

=item B<process_action>(I<action>)

Generate code for action.

=cut
sub process_action {
    my $self = shift;
    my $model = shift;
    my $action = shift;

    my $template;
    my $out;

    if (defined($action->get_name)) {
        $template = File::Spec->catfile('action',  lc($action->get_name));
        if (!$self->is_template("$template.hpp.tt")) {
            die("don't know what to do with action '" . $action->get_name . "'\n");
        }
    } else {
        die("unnamed action\n");
    }

    $out = File::Spec->catfile('src', 'model', 'action', 'Action' . $action->get_id);
    $self->process_templates($template, { 'action' => $action, 'model' => $model }, $out);

    $template = 'action_coord';
    $out = File::Spec->catfile('src', 'model', 'action', 'ActionCoord' . $action->get_id);
    $self->process_templates($template, { 'action' => $action, 'model' => $model }, $out);
}

=item B<process_client>(I<model>, I<client>)

Generate code for client program.

=over 4

=item I<model> Model, as L<Bi::Model> object.

=item I<client> Client program, as L<Bi::Client> object.

=back

No return value.

=cut
sub process_client {
    my $self = shift;
    my $model = shift;
    my $client = shift;

    my $binary = $client->get_binary;
    my $template;
    my $out;

    if ($binary =~ /^test/) {
        $template = File::Spec->catfile('test', $binary);
    } else {
        $template = File::Spec->catfile('client', $binary);
    }
    $out = File::Spec->catfile('src', $binary);
    $self->process_templates("${template}_cpu", {
        'have_model' => defined $model,
        'model' => $model,
        'client' => $client
    }, "${out}_cpu");
    $self->process_templates("${template}_gpu", {
        'have_model' => defined $model,
        'model' => $model,
        'client' => $client
    }, "${out}_gpu");
    
}

=item B<process_templates>(I<template_name>, I<vars>, I<output_name>)

Process all template files which have the given name, plus a file extension
of C<.hpp.tt>, C<.cpp.tt>, C<.cuh.tt> or C<.cu.tt>.

=cut
sub process_templates {
    my $self = shift;
    my $template_name = shift;
    my $vars = shift;
    my $output_name = shift;
    
    my @ext = ('hpp', 'cpp', 'cu', 'cuh');
    my $ext;
    
    foreach $ext (@ext) {
        if ($self->is_template("$template_name.$ext.tt")) {
            $self->process_template("$template_name.$ext.tt", $vars, "$output_name.$ext");
        }   
    }
}

=back

=head1 CLASS METHODS

=over 4

=item B<to_cpp>(I<expr>)

C++ expression filter.

=cut
sub to_cpp {
    my $expr = shift;
    if (ref($expr) eq 'ARRAY') {
    	return join('', @$expr);
    } else {
	    return Bi::Visitor::ToCpp->evaluate($expr);
    }
}

=item B<to_ascii>(I<expr>)

ASCII expression filter.

=cut
sub to_ascii {
    my $expr = shift;
    if (ref($expr) eq 'ARRAY') {
    	return join('', @$expr);
    } else {
        return Bi::Visitor::ToAscii->evaluate($expr);
    }
}

=item B<to_camel_case>(I<name>)

Converts a name_with_multiple_words to a nameWithMultipleWords.

=cut
sub to_camel_case {
    my $str = shift;
    
    $str =~ s/_(\w)/uc($1)/eg;
    $str =~ s/_$//;
    
    return $str;
}

=item B<to_typetree>(I<types>)

Creates a type tree definition from an array ref of blocks or actions.

=cut
sub to_typetree {
    my $types = shift;
        
    if (!defined($types)) {
        $types = [];
    } elsif (ref($types) ne 'ARRAY') {
        $types = [ $types ];
    }

    my $class_name;
    my $str;
    my $size = scalar(@$types);
            
    if ($size == 0) {
        $str = "NULL_NODE\n";
    } elsif ($size == 1) {
        if ($types->[0]->isa('Bi::Action')) {
            $class_name = 'Action' . $types->[0]->get_id;
        } elsif ($types->[0]->isa('Bi::Block')) {
            $class_name = 'Block' . $types->[0]->get_id;
        } else {
            assert ($types->[0]->isa('Bi::Model::Var')) if DEBUG;
            $class_name = 'Var' . $types->[0]->get_name;
        }
        $str = "LEAF_NODE(1, $class_name)\n";
    } else {
        my $upsize = 2**ceil(log($size)/log(2));
        my @leftpart = @{$types}[0..($upsize/2 - 1)];
        my @rightpart = @{$types}[($upsize/2)..($size - 1)];
        
        $str = "BEGIN_NODE(1)\n";
        $str .= to_typetree(\@leftpart);
        $str .= "JOIN_NODE\n";
        $str .= to_typetree(\@rightpart);
        $str .= "END_NODE\n";
    }
    return $str;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
