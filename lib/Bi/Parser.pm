=head1 NAME

Bi::Parser - parse LibBi model specification and construct model.

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

use parent 'Parse::Bi';
use warnings;
use strict;

use Carp::Assert;
use FindBin qw($Bin);
use IO::File;
use File::Spec;

use Parse::Bi;
use Parse::Lex;
use Bi::Model;
use Bi::Model::Spec;
use Bi::Expression;
use Bi::Visitor::Standardiser;

our @LEX_TOKENS;

BEGIN {
    use Bi qw(share_file share_dir);

    # read lex
    my $lex_file = share_file('bi.lex');
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

    my $self = new Parse::Bi;
    my $lexer = new Parse::Lex(@LEX_TOKENS);
    
    $lexer->skip('\s+'); # skips whitespace
    $self->YYData->{DATA} = $lexer;
    $self->{_model} = undef; # model, will be set by end of parse
    $self->{_action} = undef; # current action, used for in-scope aliases
    $self->{_blocks} = [];  # stack of blocks, used for other in-scopes

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
    
    # error and warning handlers
  	local $SIG{__DIE__} = sub { $self->_error(@_) };
   	local $SIG{__WARN__} = sub { $self->_warn(@_) };
    
    $self->YYParse(yylex => \&_parse_lexer, yyerror => \&_parse_error);
    #, yydebug => 0x1F);
        
    return $self->get_model;
}

=item B<get_model>

Get the model constructed by the parser.

=cut
sub get_model {
    my $self = shift;
    return $self->{_model};
}

=item B<get_action>

Get the action being constructed by the parser.

=cut
sub get_action {
	my $self = shift;
	return $self->{_action};
}

=item B<top_block>

Get the block on top of the stack.

=cut
sub top_block {
    my $self = shift;
    
    my $n = scalar(@{$self->{_blocks}}) - 1;
    my $block = $self->{_blocks}->[$n];
    
    return $block;
}

=item B<push_block>

Push a new block onto the stack.

=cut
sub push_block {
    my $self = shift;

    push(@{$self->{_blocks}}, new Bi::Block);
}

=item B<pop_block>

Pop a block from the top of the stack, and return it.

=cut
sub pop_block {
    my $self = shift;
        
    return pop(@{$self->{_blocks}});
}

=item B<push_model>

Push a new model onto the stack.

=cut
sub push_model {
    my $self = shift;

    push(@{$self->{_blocks}}, new Bi::Model);
}

=item B<pop_model>

Pop a model from the top of the stack, and return it.

=cut
sub pop_model {
    my $self = shift;
        
    my $block = pop(@{$self->{_blocks}});
    assert($block->isa('Bi::Model')) if DEBUG;
    
    return $block;
}

=back

=head2 Parsing callbacks

=over 4

=item B<model>(I<spec>)

Handle model specification.

=cut
sub model {
    my $self = shift;
    my $spec = shift;

    # bless the last remaining block into a model
    my $model = $self->pop_model;

    my $name;
    my $args = [];
    my $named_args = {};
    if (defined($spec)) {
        $name = $spec->get_name;
        $args = $spec->get_args;
        $named_args = $spec->get_named_args;
    }
    $model->set_name($name);
    $model->set_args($args);
    $model->set_named_args($named_args);
    $model->validate;
    
    $self->{_model} = $model;
}

=item B<const>(I<name>, I<expr>)

Handle const specification.

=cut
sub const {
    my $self = shift;
    my $name = shift;
    my $expr = shift;

    if ($self->top_block->is_const($name)) {
        die("constant '$name' conflicts with constant of same name in same scope.\n");
    } elsif ($self->top_block->is_inline($name)) {
        die("constant '$name' conflicts with inline of same name in same scope.\n");
    } elsif ($self->top_block->is_dim($name)) {
        die("constant '$name' conflicts with dimension of same name in same scope.\n");
    } elsif ($self->top_block->is_var($name)) {
        die("constant '$name' conflicts with variable of same name in same scope.\n");
    } elsif ($self->_is_const($name)) {
        warn("constant '$name' masks earlier declaration of constant of same name.\n");
    } elsif ($self->_is_inline($name)) {
        warn("constant '$name' masks earlier declaration of inline of same name.\n");
    } elsif ($self->_is_dim($name)) {
        warn("constant '$name' masks earlier declaration of dimension of same name.\n");
    } elsif ($self->_is_var($name)) {
        warn("constant '$name' masks earlier declaration of variable of same name.\n");
    }

    my $const = new Bi::Model::Const($name, $expr);
    $self->top_block->push_const($const);
}

=item B<inline>(I<name>, I<expr>)

Handle inline specification.

=cut
sub inline {
    my $self = shift;
    my $name = shift;
    my $expr = shift;

    if ($self->top_block->is_const($name)) {
        die("inline '$name' conflicts with constant of same name in same scope.\n");
    } elsif ($self->top_block->is_inline($name)) {
        die("inline '$name' conflicts with inline of same name in same scope.\n");
    } elsif ($self->top_block->is_dim($name)) {
        die("inline '$name' conflicts with dimension of same name in same scope.\n");
    } elsif ($self->top_block->is_var($name)) {
        die("inline '$name' conflicts with variable of same name in same scope.\n");
    } elsif ($self->_is_const($name)) {
        warn("inline '$name' masks earlier declaration of constant of same name.\n");
    } elsif ($self->_is_inline($name)) {
        warn("inline '$name' masks earlier declaration of inline of same name.\n");
    } elsif ($self->_is_dim($name)) {
        warn("inline '$name' masks earlier declaration of dimension of same name.\n");
    } elsif ($self->_is_var($name)) {
        warn("inline '$name' masks earlier declaration of variable of same name.\n");
    }

    my $inline = new Bi::Model::Inline($name, $expr);
    $self->top_block->push_inline($inline);
}

=item B<dim>(I<spec>)

Handle dimension specification.

=cut
sub dim {
    my $self = shift;
    my $spec = shift;

    my $name = $spec->get_name;
    if ($self->top_block->is_const($name)) {
        die("dimension '$name' conflicts with constant of same name in same scope.\n");
    } elsif ($self->top_block->is_inline($name)) {
        die("dimension '$name' conflicts with inline of same name in same scope.\n");
    } elsif ($self->top_block->is_dim($name)) {
        die("dimension '$name' conflicts with dimension of same name in same scope.\n");
    } elsif ($self->top_block->is_var($name)) {
        die("dimension '$name' conflicts with variable of same name in same scope.\n");
    } elsif ($self->_is_const($name)) {
        warn("dimension '$name' masks earlier declaration of constant of same name.\n");
    } elsif ($self->_is_inline($name)) {
        warn("dimension '$name' masks earlier declaration of inline of same name.\n");
    } elsif ($self->_is_dim($name)) {
        warn("dimension '$name' masks earlier declaration of dimension of same name.\n");
    } elsif ($self->_is_var($name)) {
        warn("dimension '$name' masks earlier declaration of variable of same name.\n");
    }

    my $dim = new Bi::Model::Dim($spec->get_name, $spec->get_args,
        $spec->get_named_args);
    $self->top_block->push_dim($dim);
}

=item B<var>(I<type>, I<spec>)

Handle variable specification.

=cut
sub var {
    my $self = shift;
    my $type = shift;
    my $spec = shift;
    
    my $name = $spec->get_name;
    if ($self->top_block->is_const($name)) {
        die("variable '$name' conflicts with constant of same name in same scope.\n");
    } elsif ($self->top_block->is_inline($name)) {
        die("variable '$name' conflicts with inline of same name in same scope.\n");
    } elsif ($self->top_block->is_dim($name)) {
        die("variable '$name' conflicts with dimension of same name in same scope.\n");
    } elsif ($self->top_block->is_var($name)) {
        die("variable '$name' conflicts with variable of same name in same scope.\n");
    } elsif ($self->_is_const($name)) {
        warn("variable '$name' masks earlier declaration of constant of same name.\n");
    } elsif ($self->_is_inline($name)) {
        warn("variable '$name' masks earlier declaration of inline of same name.\n");
    } elsif ($self->_is_dim($name)) {
        warn("variable '$name' masks earlier declaration of dimension of same name.\n");
    } elsif ($self->_is_var($name)) {
        warn("variable '$name' masks earlier declaration of variable of same name.\n");
    }
        
    my $var = new Bi::Model::Var($type, $spec->get_name, $spec->get_dims,
         $spec->get_args, $spec->get_named_args);
    $self->top_block->push_var($var);
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

    my $block = $self->pop_block;
    my $name;
    my $args = [];
    my $named_args = {};
    if (defined($spec)) {
        $name = $spec->get_name;
        $args = $spec->get_args;
        $named_args = $spec->get_named_args;
    }

    $block->set_name($name);
    $block->set_args($args);
    $block->set_named_args($named_args);
    $block->validate;
    
    my $top_block = $self->top_block;
    if (defined $top_block) {
        $top_block->push_child($block);
    }
    
    return $block;
}

=item B<commit_block>(I<block>)

(B<deprecated> along with do..then statement)

Tag I<block> in C<do..then> statement, or top-level block, as requiring a
commit after execution. 

=cut
sub commit_block {
    my $self = shift;
    my $block = shift;
    
    warn("the do..then statement is deprecated, actions are now executed sequentially in the order given, or in parallel but equivalent to the order given.\n");
}

=item B<top_level>(I<block>)

Tag top-level block.

=cut
sub top_level {
    my $self = shift;
    my $block = shift;
    
    $block->set_top_level(1);
}

=item B<action>(I<name>, I<aliases>, I<op>, I<expr>)

Handle action specification.

=cut
sub action {
    my $self = shift;
    my $op = shift;
    my $expr = shift;

    my $action = $self->get_action;    
    $action->set_op($op);
    $action->set_right($expr);
    $action->validate;
    
    $self->top_block->push_child($action);
    $self->{_action} = undef;
    
    return $action;
}

=item B<target>(I<name>, I<aliases>)

Handle target of an action.

=cut
sub target {
	my $self = shift;
    my $name = shift;
    my $aliases = shift;
    
    if (!defined($aliases)) {
        $aliases = [];
    }
    
    # check variable
    my $var = $self->_get_var($name);
    if (!defined $var) {
        die("no such variable '$name'\n");
    }
    
    # check dimension aliases on left
    my $num_aliases = scalar(@$aliases);
    my $num_dims = scalar(@{$var->get_dims});
    if ($num_aliases > $num_dims) {
        my $plural = ($num_dims == 1) ? '' : 's';
        die("variable '$name' has $num_dims dimension$plural, but $num_aliases aliased\n");
    }
    
    my @indexes = map { new Bi::Expression::Index(new Bi::Expression::DimAliasIdentifier($_)) } @$aliases;
    my $left = new Bi::Expression::VarIdentifier($var, \@indexes);
    
    my $action = new Bi::Action;
    $action->set_aliases($aliases);
    $action->set_left($left);
    $self->{_action} = $action;
}

=item B<dtarget>(I<name>, I<aliases>)

Handle target of differential equation.

=cut
sub dtarget {
    my $self = shift;
    my $name = shift;
    my $aliases = shift;
    
    if ($name !~ /^d/) {
        die("do you mean 'd$name'?\n")
    } else {
        $name =~ s/^d//;
        $self->target($name, $aliases);
    }
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
    
    return new Bi::Model::Spec($name, $dims, $args, \%named_args);
}

=item B<dim_arg>(I<name>)

Handle a dimension argument of a variable.

=cut
sub dim_arg {
    my $self = shift;
    my $name = shift;
    
    my $dim = $self->_get_dim($name);
    if (defined $dim) {
        return $dim;
    } else {
        die("no such dimension '$name'\n");
    }
}

=item B<dim_alias>(I<name>, I<start>, I<end>)

Handle a dimension alias of an action.

=cut
sub dim_alias {
    my $self = shift;
    my $name = shift;
    my $start = shift;
    my $end = shift;

    if (defined $name) {
        if ($self->_is_var($name)) {
            die("variable name '$name' cannot be used as dimension alias\n");
        } elsif ($self->_is_const($name)) {
            die("constant name '$name' cannot be used as dimension alias\n");
        } elsif ($self->_is_inline($name)) {
            die("inline expression name '$name' cannot be used as dimension alias\n");
        }
    }
    my $range = undef;
    if (defined $start && defined $end) {
        $range = new Bi::Expression::Range(new Bi::Expression::IntegerLiteral($start), new Bi::Expression::IntegerLiteral($end));
    } elsif (defined $start) {
    	$range = new Bi::Expression::Range(new Bi::Expression::IntegerLiteral($start));
    }
    
    return new Bi::Model::DimAlias($name, $range);
}

=item B<expression>(I<root>)

Handle unnamed expression.

=cut
sub expression {
    my $self = shift;
    my $root = shift;

    return $root;
}

=item B<literal>(I<value>)

Handle numeric literal.

=cut
sub literal {
    my $self = shift;
    my $value = shift;
    
    return new Bi::Expression::Literal($value);
}

=item B<integer_literal>(I<value>)

Handle integer literal.

=cut
sub integer_literal {
    my $self = shift;
    my $value = shift;
    
    return new Bi::Expression::IntegerLiteral($value);
}

=item B<string_literal>(I<value>)

Handle string literal.

=cut
sub string_literal {
    my $self = shift;
    my $value = shift;
    
    $value = eval($value);  # gets rid of escapes, quotes etc
    
    return new Bi::Expression::StringLiteral($value);
}

=item B<identifier>(I<name>, I<indexes>)

Handle reference to identifier.

=cut
sub identifier {
    my $self = shift;
    my $name = shift;
    my $indexes = shift;
    my $ranges = shift;
    
    if ($self->_is_const($name)) {
        if (defined($indexes) || defined($ranges)) {
            die("constant '$name' is scalar\n");
        }
        return new Bi::Expression::ConstIdentifier($self->_get_const($name));
    } elsif ($self->_is_inline($name)) {
        if (defined($indexes) || defined($ranges)) {
            die("inline expression '$name' is scalar\n");
        }
        return new Bi::Expression::InlineIdentifier($self->_get_inline($name));
    } elsif ($self->_is_var($name)) {
        my $var = $self->_get_var($name);
        if (defined($indexes) && @$indexes > 0 && @$indexes != @{$var->get_dims}) {
            my $plural1 = (@{$var->get_dims} == 1) ? '' : 's';
            my $plural2 = (@$indexes == 1) ? '' : 's'; 
            die("variable '" . $name . "' extends along " .
                scalar(@{$var->get_dims}) . " dimension$plural1, but " . scalar(@$indexes) .
                " index$plural2 given\n");
        }
        if (defined($ranges) && @$ranges > 0 && @$ranges != @{$var->get_dims}) {
            my $plural1 = (@{$var->get_dims} == 1) ? '' : 's';
            my $plural2 = (@$ranges == 1) ? '' : 's'; 
            die("variable '" . $name . "' extends along " .
                @{$var->get_dims} . " dimension$plural1, but " . scalar(@$ranges) .
                " range$plural2 given\n");
        }
        return new Bi::Expression::VarIdentifier($var, $indexes, $ranges);
    } elsif (defined($indexes) || defined($ranges)) {
        die("no variable, constant or inline expression named '$name'\n");
    } elsif (defined $self->get_action) {
    	my $alias = $self->get_action->get_alias($name);
    	if (defined $alias) {
    		return new Bi::Expression::DimAliasIdentifier($alias);
    	} else {
            die("no variable, constant, inline expression or dimension alias named '$name'\n");
    	}
    } else {
        die("no variable, constant, inline expression or dimension alias named '$name'\n");
    }
}

=item B<index>(I<index>)

Handle dimension index.

=cut
sub index {
    my $self = shift;
    my $index = shift;
    
    return new Bi::Expression::Index($index);
}

=item B<range>(I<start>, I<end>)

Handle dimension range.

=cut
sub range {
    my $self = shift;
    my $start = shift;
    my $end = shift;
    
    return new Bi::Expression::Range($start, $end);
}

=item B<function>(I<name>, I<args>)

Handle function in expression.

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

    return new Bi::Expression::Function($name, $args, \%named_args);
}

=item B<parens>(I<expr>)

Handle parentheses in expression.

=cut
sub parens {
    my $self = shift;
    my $expr = shift;
    
    return $expr;
}

=item B<unary_operator>(I<op>, I<expr>)

Handle unary operator in expression.

=cut
sub unary_operator {
    my $self = shift;
    my $op = shift;
    my $expr = shift;
    
    return new Bi::Expression::UnaryOperator($op, $expr);
}

=item B<binary_operator>(I<expr1>, I<op>, I<expr2>)

Handle binary operator in expression.

=cut
sub binary_operator {
    my $self = shift;
    my $expr1 = shift;
    my $op = shift;
    my $expr2 = shift;
    
    return new Bi::Expression::BinaryOperator($expr1, $op, $expr2);
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
    
    if ($expr1->get_shape->get_count > 0) {
        die("conditional in ternary operator must be scalar\n");
    }    
    return new Bi::Expression::TernaryOperator($expr1, $op1, $expr2, $op2, $expr3);
}

=back

=head2 State handling

=over 4

=item B<push_scope>(I<item>)

Create and push a new scope onto the stack, initialised with I<item>.

=cut
sub push_scope {
    my $self = shift;
    my $item = shift;    
}

=item B<pop_scope>

Pop a scope from the stack.

=cut
sub pop_scope {
    my $self = shift;
    
}

=item B<add_to_scope>(I<item>)

Add an item to the current scope.

=cut
sub add_to_scope {
    my $self = shift;
    my $item = shift;
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

=item B<_is_const>(I<name>)

Is there a constant called I<name> in the current scope?

=cut
sub _is_const {
    my $self = shift;
    my $name = shift;
    
    return $self->_is_item($name, \&Bi::Block::is_const);
}

=item B<_get_const>(I<name>)

Get the constant called I<name> in the current scope, or undef if no such
constant exists.

=cut
sub _get_const {
    my $self = shift;
    my $name = shift;
    
    return $self->_get_item($name, \&Bi::Block::get_const);
}

=item B<_is_inline>(I<name>)

Is there an inline expression called I<name> in the current scope?

=cut
sub _is_inline {
    my $self = shift;
    my $name = shift;
    
    return $self->_is_item($name, \&Bi::Block::is_inline);
}

=item B<_get_inline>(I<name>)

Get the inline expression called I<name> in the current scope, or undef if no
such inline expression exists.

=cut
sub _get_inline {
    my $self = shift;
    my $name = shift;
    
    return $self->_get_item($name, \&Bi::Block::get_inline);
}

=item B<_is_dim>(I<name>)

Is there a dimension called I<name> in the current scope?

=cut
sub _is_dim {
    my $self = shift;
    my $name = shift;
    
    return $self->_is_item($name, \&Bi::Block::is_dim);
}

=item B<_get_dim>(I<name>)

Get the dimension called I<name> in the current scope, or undef if no
such dimension exists.

=cut
sub _get_dim {
    my $self = shift;
    my $name = shift;
    
    return $self->_get_item($name, \&Bi::Block::get_dim);
}

=item B<_is_var>(I<name>)

Is there a variable called I<name> in the current scope?

=cut
sub _is_var {
    my $self = shift;
    my $name = shift;
    
    return $self->_is_item($name, \&Bi::Block::is_var);
}

=item B<_get_var>(I<name>)

Get the variable called I<name> in the current scope, or undef if no
such variable exists.

=cut
sub _get_var {
    my $self = shift;
    my $name = shift;
    
    return $self->_get_item($name, \&Bi::Block::get_var);
}

=item B<_is_item>(I<name>, I<subref>)

=cut
sub _is_item {
    my $self = shift;
    my $name = shift;
    my $subref = shift;
    
    my $result = 0;
    foreach my $block (reverse @{$self->{_blocks}}) {
        $result = $result || &$subref($block, $name);
    }
    return $result;
}

=item B<_is_item>(I<name>, I<subref>)

=cut
sub _get_item {
    my $self = shift;
    my $name = shift;
    my $subref = shift;
    
    my $result = undef;
    foreach my $block (reverse @{$self->{_blocks}}) {
        $result = &$subref($block, $name);
        if (defined $result) {
            return $result;
        }
    }
    return $result;
}

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

=item B<_error>

Other error routine.

=cut
sub _error {
	my $self = shift;
    my $msg = shift;
    my $lexer = $self->YYData->{DATA};
    my $line = $lexer->line;
            
    # tidy up message
    chomp $msg;
    if ($msg !~ /^Error/) {
        $msg = "Error (line $line): $msg";
    }
	die("$msg\n");
}

=item B<_warn>(I<msg>)

Warning routine.

=cut
sub _warn {
	my $self = shift;
    my $msg = shift;
    my $lexer = $self->YYData->{DATA};
    my $line = $lexer->line;
            
    # tidy up message
    chomp $msg;
    $msg =~ s/^Warning: //;
    $msg = "Warning (line $line): $msg";
    
    warn("$msg\n");
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
