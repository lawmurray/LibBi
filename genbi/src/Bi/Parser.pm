=head1 NAME

Bi::Parser - parse Bi source file and construct model.

=head1 SYNOPSIS

    use Bi::Parser;
    my $parser = new Bi::Parser();
    my $model = $parser->parse;

=head1 REQUIRES

Parse::Bi Parse::Lex

=head1 METHODS

=over 4

=cut

package Bi::Parser;

use base 'Parse::Bi';
use warnings;
use strict;

use Carp::Assert;
use FindBin qw($Bin);
use IO::File;

use Parse::Bi;
use Parse::Lex;
use Bi::Model;
use Bi::Expression;
use Bi::Visitor::Standardiser;

our @LEX_TOKENS;

BEGIN {
    # read lex
    my $lex_file = "$Bin/../src/bi.lex";
    my $file;
    my $lex;

    undef local $/;
    open($file, '<', $lex_file) || die("could not open '$lex_file'\n");
    $lex = <$file>;
    close $file;
    @LEX_TOKENS = split(/\s+/, $lex);
}

=item B<new>

Constructor.

=cut
sub new {
    my $class = shift;

    my $self = Parse::Bi->new();
    my $lexer = Parse::Lex->new(@LEX_TOKENS);
    my $model = new Bi::Model;
    
    $lexer->skip('\s+'); # skips whitespace
    $self->YYData->{DATA} = $lexer;
    $self->{_model} = $model;

    bless $self, $class;
    return $self;
}

=item B<parse>(I<fh>)

Parse model specification from I<fh> filehandle (STDIN if undef) and return
model.

=cut
sub parse {
    my $self = shift;
    my $fh = shift;
    
    if ($fh) {
        $self->YYData->{DATA}->from($fh);
    }
    $self->YYParse(yylex => \&_parse_lexer, yyerror => \&_parse_error);
    #, yydebug => 0x1F);
    
    return $self->get_model;
}

=item B<get_model>

Get the model being constructed by the parser.

=cut
sub get_model {
    my $self = shift;
    return $self->{_model};
}

=back

=head2 Parsing callbacks

=over 4

=item B<model>(I<spec>, I<defs>)

Handle model specification.

=cut
sub model {
    my $self = shift;
    my $spec = shift;
    my $defs = shift;
    
    my $def;
    my $blocks = [];
    my $consts = [];
    my $inlines = [];
    foreach $def (@$defs) {
        if (ref($def)) {
            if ($def->isa('Bi::Model::Block')) {
                push(@$blocks, $def);
            } elsif ($def->isa('Bi::Model::Const')) {
                push(@$consts, $def);
            } elsif ($def->isa('Bi::Model::Inline')) {
                push(@$inlines, $def);
            }
        } else {
            # ignore, must be part of a comment
        }
    }

    $self->get_model->init($spec->get_name, $spec->get_args, $spec->get_named_args, $blocks, $consts, $inlines);
}

=item B<dim>(I<spec>)

Handle dimension specification.

=cut
sub dim {
    my $self = shift;
    my $spec = shift;

    my $dim = Bi::Model::Dim->new($spec->get_name, $spec->get_args, $spec->get_named_args);

    $self->get_model->add_dim($dim);
    
    return $dim;
}

=item B<vars>(I<specs>, I<class>)

Handle variable specifications.

=cut
sub vars {
    my $self = shift;
    my $specs = shift;
    my $class = shift;
    
    my $spec;
    my $var;
    my $vars = [];
    
    if (ref($specs) ne 'ARRAY') {
        $specs = [ $specs ];
    }    
    foreach $spec (@$specs) {
        $var = $class->new($spec->get_name, $spec->get_dims, $spec->get_args,
            $spec->get_named_args);
        $self->get_model->add_var($var);
        $self->append($vars, $var);
    }
    
    return $vars;
}

=item B<state>(I<specs>)

Handle state variable specification.

=cut
sub state {
    my $self = shift;
    my $specs = shift;
    
    return $self->vars($specs, 'Bi::Model::State');
}

=item B<state_aux>(I<specs>)

Handle state auxiliary variable specification.

=cut
sub state_aux {
    my $self = shift;
    my $specs = shift;
    
    return $self->vars($specs, 'Bi::Model::StateAux');
}

=item B<noise>(I<specs>)

Handle noise variable specification.

=cut
sub noise {
    my $self = shift;
    my $specs = shift;
    
    return $self->vars($specs, 'Bi::Model::Noise');
}

=item B<input>(I<specs>)

Handle input specification.

=cut
sub input {
    my $self = shift;
    my $specs = shift;
    
    return $self->vars($specs, 'Bi::Model::Input');
}

=item B<obs>(I<specs>)

Handle obs specification.

=cut
sub obs {
    my $self = shift;
    my $specs = shift;
    
    return $self->vars($specs, 'Bi::Model::Obs');
}

=item B<param>(I<specs>)

Handle param specification.

=cut
sub param {
    my $self = shift;
    my $specs = shift;
    
    return $self->vars($specs, 'Bi::Model::Param');
}

=item B<param_aux>(I<specs>)

Handle auxiliary parameter specification.

=cut
sub param_aux {
    my $self = shift;
    my $specs = shift;
    
    return $self->vars($specs, 'Bi::Model::ParamAux');
}

=item B<const>(I<name>, I<expr>)

Handle const specification.

=cut
sub const {
    my $self = shift;
    my $name = shift;
    my $expr = shift;

    my $const = Bi::Model::Const->new($name, $expr);
    $self->get_model->add_const($const);
    
    return $const;
}

=item B<inline>(I<name>, I<expr>)

Handle inline specification.

=cut
sub inline {
    my $self = shift;
    my $name = shift;
    my $expr = shift;

    my $inline = Bi::Model::Inline->new($name, $expr);
    $self->get_model->add_inline($inline);
    
    return $inline;
}

=item B<positional_arg>(I<arg>)

Handle positional argument.

=cut
sub positional_arg {
    my $self = shift;
    my $arg = shift;
    
    return $arg;
}

=item B<named_arg>(I<arg>)

Handle named argument.

=cut
sub named_arg {
    my $self = shift;
    my $name = shift;
    my $arg = shift;
    
    return [ $name, $arg ];
}

=item B<block>(I<spec>, I<defs>)

Handle block specification.

=cut
sub block {
    my $self = shift;
    my $spec = shift;
    my $defs = shift;

    my $item;
    my $actions = [];
    my $blocks = [];
    my $consts = [];
    my $inlines = [];
    foreach $item (@$defs) {
        if (ref($item)) {
            if ($item->isa('Bi::Model::Block')) {
                push(@$blocks, $item);
            } elsif ($item->isa('Bi::Model::Action')) {
                push(@$actions, $item);
            } elsif ($item->isa('Bi::Model::Const')) {
                push(@$consts, $item);
            } elsif ($item->isa('Bi::Model::Inline')) {
                push(@$inlines, $item);
            }
        } else {
            # ignore, must be part of a comment
        }
    }

    if (@$actions && @$blocks) {
        # discourage user from mixing blocks and loose actions at the same
        # level
        if (@$actions == 1) {
            $self->warning('action outside block');
        } else {
            $self->warning('actions outside block');
        }
    }

    my $name;
    my $args = [];
    my $named_args = {};
    if (defined($spec)) {
        $name = $spec->get_name;
        $args = $spec->get_args;
        $named_args = $spec->get_named_args;
    }

    my $block;
    eval {
        $block = Bi::Model::Block->new($self->get_model->next_block_id, $name, $args, $named_args, $actions, $blocks, $consts, $inlines);
    };
    if ($@) {
        $self->_error($@);
    }
    
    return $block;
}

=item B<commit_block>(I<block>)

Tag I<block> in C<do..then> clause, or top-level block, as requiring a commit
after execution. 

=cut
sub commit_block {
    my $self = shift;
    my $block = shift;
    
    $block->set_commit(1);
    
    return $block;
}

=item B<action>(I<name>, I<aliases>, I<op>, I<expr>)

Handle action specification.

=cut
sub action {
    my $self = shift;
    my $name = shift;
    my $aliases = shift;
    my $op = shift;
    my $expr = shift;
       
    if (!defined($aliases)) {
        $aliases = [];
    }
       
    my $var;
    if ($self->get_model->is_var($name)) {
        $var = $self->get_model->get_var($name);
    } else {
        $self->error("no such variable '$name'");
    }
    
    my $num_aliases = scalar(@$aliases);
    my $num_dims = $var->num_dims;
    if ($num_aliases > $num_dims) {
        my $plural = ($num_dims == 1) ? '' : 's';
        $self->error("variable '$name' has $num_dims dimension$plural, but $num_aliases aliased");
    }
    
    my $dim;
    my $offsets = [];
    foreach $dim (@$aliases) {
        push(@$offsets, Bi::Expression::Offset->new($dim, 0));
    }
    my $target = Bi::Expression::VarIdentifier->new($var, $offsets);
    
    my $action;
    eval {
        $action = new Bi::Model::Action($self->get_model->next_action_id,
            $target, $op, $expr);
    };
    if ($@) {
        $self->_error($@);
    }
    
    return $action;
}

=item B<spec>(I<name>, I<dims>, I<props>)

Handle spec (name with properties).

=cut
sub spec {
    my $self = shift;
    my $name = shift;
    my $dims = shift;
    my $args = shift;
    my $named_args = shift;
    
    my %named_args = ();
    if (defined $named_args) {
        %named_args = @$named_args;
    }
    
    return Bi::Model::Spec->new($name, $dims, $args, \%named_args);
}

=item B<dim_arg>(I<name>)

Handle variable dimension argument.

=cut
sub dim_arg {
    my $self = shift;
    my $name = shift;
    
    if ($self->get_model->is_dim($name)) {
        return $self->get_model->get_dim($name);
    } else {
        $self->error("no such dimension '$name'");
    }
}

=item B<dim_alias>(I<name>)

Handle action dimension alias.

=cut
sub dim_alias {
    my $self = shift;
    my $name = shift;

    if ($self->get_model->is_var($name)) {
        $self->error("variable name '$name' cannot be used as dimension alias");
    } elsif ($self->get_model->is_const($name)) {
        $self->error("constant name '$name' cannot be used as dimension alias");
    } elsif ($self->get_model->is_inline($name)) {
        $self->error("inline expression name '$name' cannot be used as dimension alias");
    }
    
    return $name;
}

=item B<expression>(I<root>)

Handle unnamed expression.

=cut
sub expression {
    my $self = shift;
    my $root = shift;

    return Bi::Visitor::Standardiser->evaluate($root);
}

=item B<literal>(I<value>)

Handle numeric literal.

=cut
sub literal {
    my $self = shift;
    my $value = shift;
    
    return Bi::Expression::Literal->new($value);
}

=item B<string_literal>(I<value>)

Handle string literal.

=cut
sub string_literal {
    my $self = shift;
    my $value = shift;
    
    return Bi::Expression::StringLiteral->new($value);
}

=item B<identifier>(I<name>, I<offsets>)

Handle reference to identifier.

=cut
sub identifier {
    my $self = shift;
    my $name = shift;
    my $offsets = shift;
    
    if ($self->get_model->is_const($name)) {
        if (defined($offsets)) {
            $self->error("constant '$name' is scalar");
        }
        return new Bi::Expression::ConstIdentifier($self->get_model->get_const($name));
    } elsif ($self->get_model->is_inline($name)) {
        if (defined($offsets)) {
            $self->error("inline expression '$name' is scalar");
        }
        return new Bi::Expression::InlineIdentifier($self->get_model->get_inline($name));
    } elsif ($self->get_model->is_var($name)) {
        my $var = $self->get_model->get_var($name);
        if (defined($offsets) && @$offsets > 0 && @$offsets != $var->num_dims) {
            my $plural1 = ($var->num_dims == 1) ? '' : 's';
            my $plural2 = (@$offsets == 1) ? '' : 's'; 
            $self->error("variable '" . $name . "' extends along " .
                $var->num_dims . " dimension$plural1, but " . scalar(@$offsets) .
                " offset$plural2 given");
        }
        return new Bi::Expression::VarIdentifier($var, $offsets);
    } elsif (defined($offsets)) {
        $self->error("no variable, constant or inline expression named '$name'");
    } else {
        # assume at this stage that it's a dimension alias
        return new Bi::Expression::DimAlias($name)
    }
}

=item B<offset>(I<name>, I<sign>, I<value>)

Handle dimension index.

=cut
sub offset {
    my $self = shift;
    my $alias = shift;
    my $sign = shift;
    my $value = shift;
    
    # defaults
    $sign = 1 if (!defined($sign));
    $value = 0 if (!defined($value));
    $value = int($value);
    
    if ($sign eq '+') {
        $sign = 1;
    } elsif ($sign eq '-') {
        $sign = -1;
    }

    return Bi::Expression::Offset->new($alias, $sign*$value);
}

=item B<function>(I<name>, I<args>)

Handle function use in expression.

=cut
sub function {
    my $self = shift;
    my $name = shift;
    my $args = shift;
    my $named_args = shift;

    my %named_args = ();
    if (defined $named_args) {
        %named_args = @$named_args;
    }

    return Bi::Expression::Function->new($name, $args, \%named_args);
}

=item B<parens>(I<expr>)

Handle parentheses in expression.

=cut
sub parens {
    my $self = shift;
    my $expr = shift;
    
    return Bi::Expression::Parens->new($expr);
}

=item B<unary_operator>(I<op>, I<expr>)

Handle unary operator in expression.

=cut
sub unary_operator {
    my $self = shift;
    my $op = shift;
    my $expr = shift;
    
    return Bi::Expression::UnaryOperator->new($op, $expr);
}

=item B<binary_operator>(I<expr1>, I<op>, I<expr2>)

Handle binary operator in expression.

=cut
sub binary_operator {
    my $self = shift;
    my $expr1 = shift;
    my $op = shift;
    my $expr2 = shift;
    
    return Bi::Expression::BinaryOperator->new($expr1, $op, $expr2);
}

=item B<ternary_operator>(I<expr1>, I<op1>, I<expr2>, I<op2>, I<expr3>)

Handle ternary operator in expression.

=cut
sub ternary_operator {
    my $self = shift;
    my $expr1 = shift;
    my $op1 = shift;
    my $expr2 = shift;
    my $op2 = shift;
    my $expr3 = shift;
    
    if ($expr1->num_dims > 0) {
        $self->error("conditional in ternary operator must be scalar");
    }    
    return Bi::Expression::TernaryOperator->new($expr1, $op1, $expr2, $op2, $expr3);
}

=back

=head2 Utility methods

=over 4

=item B<error>(I<msg>)

Print I<msg> as error and terminate.

=cut
sub error {
    my $self = shift;
    my $msg = shift;

    my $lexer = $self->YYData->{DATA};
    my $line = $lexer->line;
    my $text = $lexer->token->text;
    
    chomp $msg;
    die("Error (line $line): $msg\n");
}

=item B<warn>(I<msg>)

Print I<msg> as warning.

=cut
sub warning {
    my $self = shift;
    my $msg = shift;

    my $lexer = $self->YYData->{DATA};
    my $line = $lexer->line;
    my $text = $lexer->token->text;
    
    chomp $msg;
    warn("Warning (line $line): $msg\n");
}

=item B<append>(I<list>, I<value>)

Append I<value> to I<list>.

=cut
sub append {
    my $self = shift;
    my $list = shift;
    my $value = shift;
 
    if (!defined($list)) {
        $list = [];
    } elsif (ref($list) ne 'ARRAY') {
        $list = [ $list ];
    }
    if (defined($value)) {
        if (ref($value) eq 'ARRAY') {
            push(@$list, @$value);     
        } else {
            push(@$list, $value);     
        }
    }
    return $list;
}

=back

=head2 Internal methods

=over 4

=item B<_parse_lexer>

Lexer routine for L<Parse::Yapp>.

=cut
sub _parse_lexer {
    my $self = shift;
    my $lexer = $self->YYData->{DATA};
    my $token;
    my $name;
    my $text;
    my $in_comment = 0;

    do {
        $token = $lexer->next;
           if ($lexer->eoi) {
            return ('', undef);
        }

        if ($token->name eq 'OP') {
            # OP used as alphanumeric placeholder in lexer only, replace now
            $name = $token->text;
            $text = $token->text;
        } else {
            $name = $token->name;
            $text = $token->text;
        }
        if ($name eq 'COMMENT_BEGIN') {
            ++$in_comment;
        } elsif ($name eq 'COMMENT_END') {
            --$in_comment;
            if ($in_comment < 0) {
                $self->_parse_error;
            }
        }
    } while ($name =~ /^COMMENT_/ || $in_comment > 0);
    
    return ($name, $text);
}

=item B<_parse_error>

Error routine for L<Parse::Yapp>.

=cut
sub _parse_error {
    my $self = shift;
    my $lexer = $self->YYData->{DATA};
    my $line = $lexer->line;
    my $text = $lexer->token->text;

    die("Error (line $line): syntax error near '$text'\n");
}

=item B<_parse_error>

Other error routine.

=cut
sub _error {
    my $self = shift;
    my $msg = shift;
    my $lexer = $self->YYData->{DATA};
    my $line = $lexer->line;
    
    chomp($msg);
    die("Error (line $line): $msg\n");
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
