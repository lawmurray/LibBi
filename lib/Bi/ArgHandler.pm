=head1 NAME

Bi::ArgHandler - generic class for handling positional and named arguments.

=head1 SYNOPSIS

Use derived class.

=head1 METHODS

=over 4

=cut

package Bi::ArgHandler;

use warnings;
use strict;

use Carp::Assert;
use Bi::Utility;
use Bi::Expression;

=item B<new>(I<args>, I<named_args>)

Constructor.

=over 4

=item I<args>

Ordered list of positional arguments to the function as L<Bi::Expression>
objects.

=item I<named_args>

Hash of named arguments to the function, keyed by name, as L<Bi::Expression>
objects.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $args = shift;
    my $named_args = shift;

    # pre-conditions
    assert(!defined($args) || ref($args) eq 'ARRAY') if DEBUG;
    assert(!defined($named_args) || ref($named_args) eq 'HASH') if DEBUG;
    if (!defined $args) {
        $args = [];
    }
    if (!defined $named_args) {
        $named_args = {};
    }

    my $self = {
        _args => $args,
        _named_args => $named_args
    };
    bless $self, $class;
   
    return $self;
}

=item B<clone>

Return a clone of the object.

=cut
sub clone {
    my $self = shift;
    
    my $clone = {
        _args => [],
        _named_args => {}
    };
    bless $clone, 'Bi::ArgHandler';
    
    my $arg;
    foreach $arg (@{$self->get_args}) {
        push(@{$clone->get_args}, $arg->clone);
    }
    foreach $arg (keys %{$self->get_named_args}) {
        $clone->get_named_args->{$arg} = $self->get_named_args->{$arg}->clone;
    }
    
    return $clone;
}

=item B<get_args>

Get the arguments of the function.

=cut
sub get_args {
    my $self = shift;
    return $self->{_args};
}

=item B<get_args>(I<args>)

Set the arguments of the function.

=cut
sub set_args {
    my $self = shift;
    my $args = shift;
    
    # pre-conditions
    assert(ref($args) eq 'ARRAY') if DEBUG;
    
    $self->{_args} = $args;
}

=item B<get_arg>(I<i>)

Get the I<i>th argument.

=cut
sub get_arg {
    my $self = shift;
    my $i = shift;
    
    $i = 0 if (!defined $i);
    assert($i >= 0 && $i < $self->num_args) if DEBUG;
    
    return $self->get_args->[$i];
}

=item B<get_named_args>

Get the named arguments of the function.

=cut
sub get_named_args {
    my $self = shift;
    return $self->{_named_args};
}

=item B<set_named_args>(I<named_args>)

Set the named arguments of the function.

=cut
sub set_named_args {
    my $self = shift;
    my $named_args = shift;
    
    # pre-conditions
    assert(ref($named_args) eq 'HASH') if DEBUG;
    map { assert($_->isa('Bi::Expression')) if DEBUG } values %$named_args;
    
    $self->{_named_args} = $named_args;
}

=item B<get_named_arg>(I<name>)

Get a named argument.

=cut
sub get_named_arg {
    my $self = shift;
    my $name = shift;
    
    assert($self->is_named_arg($name)) if DEBUG;
    
    return $self->get_named_args->{$name};
}

=item B<set_named_arg>(I<name>, I<expr>)

Set a named argument.

=cut
sub set_named_arg {
    my $self = shift;
    my $name = shift;
    my $expr = shift;
    
    assert ($expr->isa('Bi::Expression')) if DEBUG;
    
    $self->get_named_args->{$name} = $expr;
}

=item B<is_named_arg>(I<name>)

Is there a named argument of name I<name>?

=cut
sub is_named_arg {
    my $self = shift;
    my $name = shift;
    
    return exists $self->get_named_args->{$name};
}

=item B<num_args>

Get number of arguments to function.

=cut
sub num_args {
    my $self = shift;
    return int(scalar(@{$self->get_args}));
}

=item B<num_named_args>

Get number of named arguments to function.

=cut
sub num_named_args {
    my $self = shift;
    return int(scalar(keys %{$self->get_named_args}));
}

=item B<process_args>(I<spec>)

Process both positional and named arguments according to a given
specification, and translate all into named arguments. 

=over 4

=item I<spec> Array-of-hashes parameter specification. Each element of the
outer array consists of a hash with up to four key-value pairs:

=over 4

=item I<name> of the parameter (mandatory),

=item I<positional> indicating whether or not the parameter may be used
positionally (defaults to 0),

=item I<mandatory> indicating whether or not the parameter is required
(defaults to 0), and

=item I<default> giving a default value if the parameter is not supplied.

=item I<deprecated> indicating whether or not the parameter is deprecated,
and if so

=item I<message> giving a message to display, suggesting an alternative
option to the user, for example.

=back

=back

The order in which parameters appear in the I<spec> array is assumed to be the
same as the order in which they should be used in a positional context.

The following checks are made:

=over 4

=item * that all mandatory arguments are supplied as either positional or
named arguments,

=item * that where an argument is supplied as a positional argument, it is
not also provided as a named argument,

=item * that no unrecognised named arguments are given.

=back

Note that, for the purpose of checking whether a mandatory argument has been
supplied, a parameter with a default value is always considered to have been
supplied.

If any checks fail, an appropriate error message is printed and an exception
thrown. If all checks pass, the arguments of the item are all converted into
named arguments, and positional arguments emptied.

=cut
sub process_args {
    my $self = shift;
    my $specs = shift;
    
    my $name = $self->get_name;
    my @position_names;
    my %names;
    my $spec;
    my $arg;
    my $args = {};

    # map positions to names
    foreach $spec (@$specs) {
        assert(ref($spec) eq 'HASH') if DEBUG;
        assert(!exists $names{$spec->{name}}) if DEBUG;
        
        if ($spec->{positional}) {
            push(@position_names, $spec->{name});
        }
        $names{$spec->{name}} = 1;
    }
    my $n = scalar(@position_names);
    
    # handle positional arguments
    my $i = 0;
    foreach $arg (@{$self->get_args}) {
        if ($i < $n) {
            $args->{$position_names[$i]} = $arg;
            ++$i;
        } else {
            if ($n == 0) {
                die("action '$name' does not take positional arguments\n");
            } elsif ($n == 1) {
                die("action '$name' takes at most 1 positional argument\n");
            } else {
                die("action '$name' takes at most $n positional arguments\n");
            }
        }
    }
    
    # handle named arguments
    foreach $arg (keys %{$self->get_named_args}) {
        if (!exists $names{$arg}) {
            die("unrecognised named argument '$arg'\n");
        } elsif (exists $args->{$arg}) {
            die("named argument '$arg' already used as positional\n")
        } else {
            $args->{$arg} = $self->get_named_args->{$arg};
        }
    }
    
    # check mandatory arguments and apply defaults
    foreach $spec (@$specs) {
        if (!exists $args->{$spec->{name}} || !defined $args->{$spec->{name}}) {
            if (exists $spec->{default}) {
                my $literal;
                
                if (is_integer($spec->{default})) {
                    $literal = new Bi::Expression::IntegerLiteral($spec->{default});
                } elsif (is_number($spec->{default})) {
                    $literal = new Bi::Expression::Literal($spec->{default});
                } else {
                    $literal = new Bi::Expression::StringLiteral($spec->{default});
                }
                $args->{$spec->{name}} = $literal;
            } elsif ($spec->{mandatory}) {
                die("missing mandatory argument '" . $spec->{name} . "'\n");
            }
        }
    }
    
    # check deprecated arguments
    foreach $spec (@$specs) {
        if ($spec->{deprecated} && exists $args->{$spec->{name}}) {
            warn($spec->{name} . " is deprecated, " . $spec->{message} . "\n");
        }
    }
    
    # modify action
    $self->set_args([]);
    $self->set_named_args($args);
}

=item B<ensure_const>(I<name>)

Ensure that named argument I<name> is a constant expression.

=cut
sub ensure_const {
    my $self = shift;
    my $name = shift;
    
    if (!(!$self->is_named_arg($name) || $self->get_named_arg($name)->is_const)) {
        my $action = $self->get_name;
        die("argument '$name' to action '$action' must be a constant expression\n");
    }
}

=item B<ensure_common>(I<name>)

Ensure that named argument I<name> is a common expression.

=cut
sub ensure_common {
    my $self = shift;
    my $name = shift;
    
    if (!(!$self->is_named_arg($name) || $self->get_named_arg($name)->is_common)) {
        my $action = $self->get_name;
        die("argument '$name' to action '$action' must be a common expression\n");
    }
}

=item B<ensure_scalar>(I<name>)

Ensure that named argument I<name> is a scalar.

=cut
sub ensure_scalar {
    my $self = shift;
    my $name = shift;
    
    if (!(!$self->is_named_arg($name) || $self->get_named_arg($name)->is_scalar)) {
        my $action = $self->get_name;
        die("argument '$name' to action '$action' must be a scalar\n");
    }
}

=item B<ensure_vector>(I<name>)

Ensure that named argument I<name> is a vector.

=cut
sub ensure_vector {
    my $self = shift;
    my $name = shift;
    
    if (!(!$self->is_named_arg($name) || $self->get_named_arg($name)->is_vector)) {
        my $action = $self->get_name;
        die("argument '$name' to action '$action' must be a vector\n");
    }
}

=item B<ensure_matrix>(I<name>)

Ensure that named argument I<name> is a matrix.

=cut
sub ensure_matrix {
    my $self = shift;
    my $name = shift;
    
    if (!(!$self->is_named_arg($name) || $self->get_named_arg($name)->is_matrix)) {
        my $action = $self->get_name;
        die("argument '$name' to action '$action' must be a matrix\n");
    }
}

=item B<ensure_num_dims>(I<name>, I<num>)

Ensure that named argument I<name> has at most I<num> dimensions.

=cut
sub ensure_num_dims {
    my $self = shift;
    my $name = shift;
    my $num = shift;
    
    if (!(!$self->is_named_arg($name) || $self->get_named_arg($name)->get_shape->get_count <= $num)) {
        my $action = $self->get_name;
        my $plural = '';
        if ($num == 1) {
            $plural = 's';
        }
        die("argument '$name' to action '$action' may have at most $num dimension$plural\n");
    }
}

=item B<accept>(I<visitor>, ...)

Accept visitor. This propagates the visitor over arguments, it does not
propagate over the object itself, which is left to the derived class to
handle.

=cut
sub accept {
    my $self = shift;
    my $visitor = shift;
    my @args = @_;

    for (my $i = 0; $i < @{$self->get_args}; ++$i) {
        $self->get_args->[$i] = $self->get_args->[$i]->accept($visitor, @args);
    }
    foreach my $key (sort keys %{$self->get_named_args}) {
	    $self->get_named_args->{$key} = $self->get_named_args->{$key}->accept($visitor, @args);
    }
}

=item B<equals>(I<obj>)

Does object equal C<obj>?

=cut
sub equals {
    my $self = shift;
    my $obj = shift;
    
    my $equals = 
        $self->num_args == $obj->num_args &&
        $self->num_named_args == $obj->num_named_args &&
        Bi::Utility::equals($self->get_args, $obj->get_args);
    if ($equals) {
        my $key;
        foreach $key (keys %{$self->get_named_args}) {
            $equals = $equals && $obj->is_named_arg($key) && $self->get_named_arg($key)->equals($obj->get_named_arg($key));
        }
    }
    
    return $equals;
}

=back

=head1 CLASS METHODS

=over 4

=item B<is_number>(I<value>)

Is I<value> a number?

=cut
sub is_number {
    my $value = shift;
    
    # regexps are taken from bi.lex
    my $res = 0;
    $res = $res || $value =~ /^[0-9]+[Ee][+-]?[0-9]+$/;
    $res = $res || $value =~ /^[0-9]+\.[0-9]*([Ee][+-]?[0-9]+)?$/;
    $res = $res || $value =~ /^[0-9]*\.[0-9]+([Ee][+-]?[0-9]+)?$/;
    $res = $res || $value =~ /^[0-9]+$/;
    
    return $res;
}

=item B<is_integer>(I<value>)

Is I<value> an integer?

=cut
sub is_integer {
    my $value = shift;
    
    return $value =~ /^[0-9]+$/;
}

1;

=back

=head1 SEE ALSO

L<Bi::Expression>

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
