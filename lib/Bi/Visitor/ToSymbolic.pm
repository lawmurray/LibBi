=head1 NAME

Bi::Visitor::ToSymbolic - visitor for translating expression into a form
suitable for L<Math::Symbolic>.

=head1 SYNOPSIS

    use Bi::Visitor::ToSymbolic;
    $visitor = Bi::Visitor::ToSymbolic->new;
    
    my $symb = $visitor->expr2symb($expr);
    my $expr = $visitor->symb2expr($symb);

=head1 INHERITS

L<Bi::Visitor>

=head1 METHODS

=over 4

=cut

package Bi::Visitor::ToSymbolic;

use parent 'Bi::Visitor';
use warnings;
use strict;

use Carp::Assert;
use Math::Symbolic qw(:constants);

use Bi::Expression;

our $Op_Types;

BEGIN {
    # override Math::Symbolic operator prefixes to be compatible with
    # Bi::Expression
    $Op_Types = \@Math::Symbolic::Operator::Op_Types;
    
    #$Op_Types->[B_EXP]->{infix_string} = undef;
    $Op_Types->[B_EXP]->{prefix_string} = 'pow';
    #$Op_Types->[U_MINUS]->{infix_string} = undef;
    $Op_Types->[U_MINUS]->{prefix_string} = '-';
}

=item B<new>

Constructor.

=cut
sub new {
    my $class = shift;
    my $self = {
        _substitutes => {}
    };
    bless $self, $class;
    return $self;
}

=item B<expr2symb>(I<expr>)

Converts a L<Bi::Expression> object to an expression tree for
L<Math::Symbolic>. This involves some variable name translations, which are
stored internally to accurately recover L<Bi::Expression> objects later using
B<symb2expr>.

=over 4

=item I<expr> L<Bi::Expression> object.

=back

Returns a L<Math::Symbolic> expression tree.

=cut
sub expr2symb {
    my $self = shift;
    my $expr = shift;
    
    my $args = [];
    $expr->accept($self, $args);    
    return pop(@$args);
}

=item B<symb2expr>(I<symb>)

Converts an expression tree for L<Math::Symbolic> to a L<Bi::Expression>
object. This involves some variable name translations, which are recovered
from a previous call to B<expr2symb>.

=over 4

=item I<symb> L<Math::Symbolic> expression tree.

=back

Returns a L<Bi::Expression> object.

=cut
sub symb2expr {
    my $self = shift;
    my $symb = shift;
    my $expr;
    
    if ($symb->isa('Math::Symbolic::Operator')) {
        if ($symb->arity == 1) {
            my $op = $Op_Types->[$symb->type]->{prefix_string};
            my $expr1 = $self->symb2expr($symb->op1);
            if ($op =~ /\w+/) {
                # actually a function
                $expr = new Bi::Expression::Function($op, [ $expr1 ]);
            } else {
                # an operator
                # error here using new Bi::Expression::UnaryOperator...
                $expr = Bi::Expression::UnaryOperator->new($op, $expr1);
            }
        } elsif ($symb->arity == 2) {
            my $op = $Op_Types->[$symb->type]->{infix_string} ||
                    $Op_Types->[$symb->type]->{prefix_string};
            if ($op eq 'log') {
                # remove second argument (base) to make unary function
                if ($symb->op2->to_string eq 'EULER') {
                    $expr = new Bi::Expression::Literal(1.0);
                } else {
                    my $expr1 = $self->symb2expr($symb->op1);
                    $expr = new Bi::Expression::Function($op, [ $expr1 ]);
                }
            } else {
                if ($op eq '^') {
                    if ($symb->op1->to_string eq 'EULER') {
                        # override this with math exp()
                        $op = 'exp';
                        my $expr2 = $self->symb2expr($symb->op2);
                        $expr = new Bi::Expression::Function($op, [ $expr2 ]);
                    } else {
                        # override this with math pow()
                        $op = 'pow';
                        my $expr1 = $self->symb2expr($symb->op1);
                        my $expr2 = $self->symb2expr($symb->op2);
                        $expr = new Bi::Expression::Function($op, [ $expr1, $expr2 ]);
                    }
                } else {
                    my $expr1 = $self->symb2expr($symb->op1);
                    my $expr2 = $self->symb2expr($symb->op2);
                            
                    if ($op =~ /\w+/) {
                        # actually a function
                        $expr = new Bi::Expression::Function($op, [ $expr1, $expr2 ]);
                    } else {
                        # an operator, parens ensure correct precedence
                        $expr = new Bi::Expression::BinaryOperator($expr1, $op, $expr2);
                    }
                }
            }
        } else {
            die('unsupported symbolic tree operator');
        }
    } elsif ($symb->isa('Math::Symbolic::Constant')) {
        $expr = new Bi::Expression::Literal($symb->value);
    } elsif ($symb->isa('Math::Symbolic::Variable')) {
        if ($symb->name eq 'EULER') {
            $expr = new Bi::Expression::Literal(exp(1.0));
        } elsif ($symb->name eq 'PI') {
            $expr = new Bi::Expression::Literal(cos(1.0));
        } else {
            $expr = $self->_recover($symb->name);
        }
    } else {
        die("unsupported expression type\n");
    }
    
    return $expr;
}

=item B<visit_after>(I<node>, I<args>)

Visit node.

=cut
sub visit_after {
    my $self = shift;
    my $node = shift;
    my $args = shift;
    my $symb;
    
    if ($node->isa('Bi::Expression::BinaryOperator')) {
        my @symbs = splice(@$args, -2);
        my $op = $node->get_op;
        $symb = new Math::Symbolic::Operator($op, @symbs);
    } elsif ($node->isa('Bi::Expression::Function')) {
        my $num_args = $node->num_args + $node->num_named_args;
        my @symbs = splice(@$args, -$num_args, $num_args);
        my $name = $node->get_name;
        if ($name eq 'pow') {
            # convert to ^ operator
            $name = '^';
        } elsif ($name eq 'log') {
            # binary form, including argument for base
            unshift(@symbs, 'EULER');
        } elsif ($name eq 'exp') {
            # convert to ^ operator
            $name = '^';
            unshift(@symbs, 'EULER');
        } elsif ($name eq 'sqrt') {
            # convert to ^ operator
            $name = '^';
            push(@symbs, new Math::Symbolic::Constant(0.5));
        } elsif ($name eq 'abs') {
        	# convert to square then square-root
        	$name = '^';
        	push(@symbs, new Math::Symbolic::Constant(2.0));
        	@symbs = new Math::Symbolic::Operator($name, @symbs);
        	push(@symbs, new Math::Symbolic::Constant(0.5));
        }
        $symb = new Math::Symbolic::Operator($name, @symbs);
    } elsif ($node->isa('Bi::Expression::ConstIdentifier')) {
        my $name = $self->_substitute($node);
        $symb = new Math::Symbolic::Variable($name);
    } elsif ($node->isa('Bi::Expression::InlineIdentifier')) {
        $symb = $self->expr2symb($node->get_inline->get_expr);
    } elsif ($node->isa('Bi::Expression::VarIdentifier')) {
        my $name = $self->_substitute($node);
        $symb = new Math::Symbolic::Variable($name);
    } elsif ($node->isa('Bi::Expression::DimAliasIdentifier')) {
        $symb = $self->_substitute($node);
    } elsif ($node->isa('Bi::Expression::Literal')) {
        $symb = new Math::Symbolic::Constant($node->get_value);
    } elsif ($node->isa('Bi::Expression::IntegerLiteral')) {
        $symb = new Math::Symbolic::Constant($node->get_value);
    } elsif ($node->isa('Bi::Expression::StringLiteral')) {
        die("cannot convert string literals to symbolic\n");
    } elsif ($node->isa('Bi::Expression::Index')) {
        pop(@$args);
    	# handled by owning Bi::Expression::VarIdentifier
    } elsif ($node->isa('Bi::Expression::Range')) {
        pop(@$args);
    	# handled by owning Bi::Expression::VarIdentifier
    } elsif ($node->isa('Bi::Expression::TernaryOperator')) {
        die('cannot convert ternary operator ?: to symbolic');
    } elsif ($node->isa('Bi::Expression::UnaryOperator')) {
        $symb = pop(@$args);
        my $op = $node->get_op;
        if ($op eq '-') {
            # Math::Symbolic interprets this as a binary subtract otherwise
            $op = 'neg';
        }
        $symb = new Math::Symbolic::Operator($op, $symb);
    } else {
        die("unrecognised node in expression, type " . ref($node));
    }
    
    if (defined $symb) {
        push(@$args, $symb);
    }
    
    return $node;
}

=item B<_substitute>(I<expr>)

Replace an expression with a unique name. If a name for the same
expression has already been generated, it is returned, otherwise a new name
is generated.

=cut
sub _substitute {
    my $self = shift;
    my $expr = shift;
    
    my @names = keys %{$self->{_substitutes}};
    my @exprs = values %{$self->{_substitutes}};
    my $i;
    for ($i = 0; $i < @exprs; ++$i) {
        if ($exprs[$i]->equals($expr)) {
            return $names[$i];
        }
    }
    
    # not found
    my $name = "var$i";
    $self->{_substitutes}->{$name} = $expr;

    return $name;
}

=item B<_recover>(I<name>)

Recover expression from substituted name.

=cut
sub _recover {
    my $self = shift;
    my $name = shift;
    
    assert (exists $self->{_substitutes}->{$name}) if DEBUG;
    
    return $self->{_substitutes}->{$name};
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

