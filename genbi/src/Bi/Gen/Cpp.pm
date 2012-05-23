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

use base 'Bi::Gen';
use warnings;
use strict;

use Carp::Assert;
use FindBin qw($Bin);
use File::Spec;
use POSIX qw(ceil);

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

    my $ttdir = File::Spec->catdir($Bin, '..', 'tt', 'cpp');
    my $self = Bi::Gen->new($ttdir, $outdir);
    $self->get_tt->context->define_vmethod('hash', 'to_cpp', \&to_cpp);
    $self->get_tt->context->define_vmethod('hash', 'to_ascii', \&to_ascii);
    $self->get_tt->context->define_vmethod('array', 'to_typetree', \&to_typetree);
    $self->get_tt->context->define_filter('to_camel_case', \&to_camel_case);
    
    bless $self, $class;
    return $self;
}

=item B<gen>(I<model>)

Generate code for model.

=cut
sub gen {
    my $self = shift;
    my $model = shift;
    my $out;
    
    # pre-condition
    assert($model->isa('Bi::Model')) if DEBUG;

    # model    
    $out = 'src/model/Model' . ucfirst($model->get_name);
    $self->process_templates('model', { 'model' => $model }, $out);

    # dimensions
    foreach my $dim (@{$model->get_dims}) {
        $self->process_dim($model, $dim);
    }   
    
    # variables
    foreach my $var (@{$model->get_vars}) {
        $self->process_var($model, $var);
    }    

    # blocks
    foreach my $block (@{$model->get_blocks}) {
        $self->process_block($model, $block);
    }
}

=item B<process_dim>(I<dim>)

Generate code for dimension.

=cut
sub process_dim {
    my $self = shift;
    my $model = shift;
    my $dim = shift;

    my $template = 'dim';
    my $out = 'src/model/dim/Dim' . $dim->get_name;
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
    my $out = 'src/model/var/Var' . $var->get_name;
    $self->process_templates($template, { 'var' => $var, 'model' => $model }, $out);    
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
    my $subblock;
    my $action;

    if (defined($block->get_name)) {
        $template = 'block/' . lc($block->get_name);
        if (!$self->is_template("$template.hpp.tt")) {
            die("don't know what to do with block '" . $block->get_name . "'\n");
        }
    } else {
        $template = 'block/eval';
    }
    
    # block     
    $out = 'src/model/block/Block' . $block->get_id;
    $self->process_templates($template, { 'block' => $block, 'model' => $model }, $out);

    # sub-blocks
    foreach $subblock (@{$block->get_blocks}) {
        $self->process_block($model, $subblock);
    }   
    
    # actions
    foreach $action (@{$block->get_actions}) {
        $self->process_action($model, $action);
    }
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
        $template = 'action/' . lc($action->get_name);
    } else {
        $template = 'action/eval';
    }

    $out = 'src/model/action/Action' . $action->get_id;
    if ($self->is_template("$template.hpp.tt")) {
        $self->process_templates($template, { 'action' => $action, 'model' => $model }, $out);
    } else {
        $template = 'action/default';
        $self->process_templates($template, { 'action' => $action, 'model' => $model }, $out);
    }
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

    $template = "client/$binary";
    $out = "src/$binary";
    $self->process_templates("${template}_cpu",
        { 'model' => $model, 'client' => $client }, "${out}_cpu");
    $self->process_templates("${template}_gpu",
        { 'model' => $model, 'client' => $client }, "${out}_gpu");
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
    return Bi::Visitor::ToCpp->evaluate($expr);
}

=item B<to_ascii>(I<expr>)

ASCII expression filter.

=cut
sub to_ascii {
    my $expr = shift;    
    return Bi::Visitor::ToAscii->evaluate($expr);
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
        
    assert (!defined($types) || ref($types) eq 'ARRAY') if DEBUG;
    if (!defined($types)) {
        $types = [];
    }

    my $class_name;
    my $str;
    my $size = scalar(@$types);
            
    if ($size == 0) {
        $str = "NULL_NODE\n";
    } elsif ($size == 1) {
        if ($types->[0]->isa('Bi::Model::Action')) {
            $class_name = 'Action' . $types->[0]->get_id;
        } elsif ($types->[0]->isa('Bi::Model::Block')) {
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
