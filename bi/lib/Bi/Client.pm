=head1 NAME

Bi::Client - generic client program.

=head1 SYNOPSIS

    use Bi::Client;
    my $client = new Bi::Client($cmd, $builddir);
    $client->process_args;
    $client->exec;  # if successful, never returns

=head1 COMMAND LINE

Common command line options are documented in the developer guide.

=cut

package Bi::Client;

use warnings;
use strict;

use Carp::Assert;
use Cwd qw(abs_path);
use Getopt::Long qw(:config pass_through no_auto_abbrev no_ignore_case);

# options specific to execution
our @EXEC_OPTIONS = (
    {
      name => 'with-gdb',
      type => 'bool',
      default => 0
    },
    {
      name => 'with-valgrind',
      type => 'bool',
      default => 0
    },
    {
      name => 'with-cuda-gdb',
      type => 'bool',
      default => 0
    },
    {
      name => 'with-cuda-memcheck',
      type => 'bool',
      default => 0
    }
);

# options to pass to client program
our @CLIENT_OPTIONS = (
    {
      name => 'seed',
      type => 'int',
      default => 0
    },
    {
      name => 'threads',
      type => 'int',
      default => 0
    },
    {
      name => 'enable-timing',
      type => 'bool',
      default => 0
    },
    {
      name => 'enable-output',
      type => 'bool',
      default => 1
    },
    {
      name => 'with-gperftools',
      type => 'bool',
      default => 0
    },
    {
      name => 'gperftools-file',
      type => 'string',
      default => 'pprof.prof'
    },
    {
      name => 'with-mpi',
      type => 'bool',
      default => 0
    },
    {
      name => 'mpi-np',
      type => 'int',
    },
    {
      name => 'mpi-npernode',
      type => 'int',
    },
    {
      name => 'transform-extended',
      type => 'bool',
      default => 0
    },
    {
      name => 'transform-param-to-state',
      type => 'bool',
      default => 0
    },
    {
      name => 'transform-obs-to-state',
      type => 'bool',
      default => 0
    },
    {
      name => 'transform-initial-to-param',
      type => 'bool',
      default => 0
    }
);

=head1 METHODS

=over 4

=item B<new>(I<cmd>, I<builddir>, I<verbose>)

Constructor.

=over 4

=item I<cmd>

The command.

=item I<builddir>

The build directory

=item I<verbose>

Is verbosity enabled?

=back

Returns the new object, which will be of a class derived from L<Bi::Client>,
not of type L<Bi::Client> directly. The type of the object is inferred from
I<cmd>.

=cut
sub new {
    my $class = shift;
    my $cmd = shift;
    my $builddir = shift;
    my $verbose = shift;

    $builddir = abs_path($builddir);
    my $self = {
        _builddir => $builddir,
        _verbose => $verbose,
        _binary => '',
        _params => [ @CLIENT_OPTIONS ],
        _exec_params => [ @EXEC_OPTIONS ],
        _args => {},
        _exec_args => {},
    };
    
    # look up appropriate class name
    $class = "Bi::Client::$cmd";
    unless (eval("require $class")) {
        $class = "Bi::Test::$cmd";
        eval("require $class") || die("don't know what to do with command '$cmd'\n");
    }
    bless $self, $class;
    
    # override default gperftools output file
    foreach my $param (@{$self->get_params}) {
    	if ($param->{name} eq 'gperftools-file') {
    		$param->{default} = "$cmd.prof";
    	}
    }
        
    # common arguments
    $self->init;
    return $self;
}

=item B<get_binary>

Get the name of the binary associated with this client program.

=cut
sub get_binary {
    my $self = shift;
    return $self->{_binary};
}

=item B<is_cpp>

Is this a C++ client?

=cut
sub is_cpp {
    return 1;
}

=item B<needs_model>

Does this client need a model?

=cut
sub needs_model {
    return 1;
}

=item B<get_params>

Get formal parameters to be passed to the client program as a hashref. 

=cut
sub get_params {
    my $self = shift;
    return $self->{_params};
}

=item B<get_exec_params>

Get formal parameters not to be passed to the client program as a hashref.
Get formal parameters of the client program as a hash. 

=cut
sub get_exec_params {
    my $self = shift;
    return $self->{_exec_params};
}

=item B<get_args>

Get arguments to the client program as a hash. These are available after a
call to B<process_args>.

=cut
sub get_args {
    my $self = shift;
    
    return $self->{_args};
}

=item B<get_exec_args>

Get arguments not to be passed to the client program as a hashref. These are
available after a call to B<process_args>.

=cut
sub get_exec_args {
    my $self = shift;
    
    return $self->{_exec_args};
}

=item B<is_named_arg>(I<name>)

Is there a named argument of name I<name>?

=cut
sub is_named_arg {
    my $self = shift;
    my $name = shift;
    
    return exists $self->get_args->{$name} && defined $self->get_args->{$name};
}

=item B<is_named_exec_arg>(I<name>)

Is there a named argument of name I<name>?

=cut
sub is_named_exec_arg {
    my $self = shift;
    my $name = shift;
    
    return exists $self->get_exec_args->{$name} && defined $self->get_exec_args->{$name};
}

=item B<get_named_arg>(I<name>)

Get named argument.

=cut
sub get_named_arg {
    my $self = shift;
    my $name = shift;
    
    return $self->get_args->{$name};
}

=item B<set_named_arg>(I<name>, I<value>)

Set named argument.

=cut
sub set_named_arg {
    my $self = shift;
    my $name = shift;
    my $value = shift;
    
    $self->get_args->{$name} = $value;
}

=item B<get_named_exec_arg>(I<name>)

Get named argument.

=cut
sub get_named_exec_arg {
    my $self = shift;
    my $name = shift;
    
    return $self->get_exec_args->{$name};
}

=item B<set_named_exec_arg>(I<name>, I<value>)

Set named argument.

=cut
sub set_named_exec_arg {
    my $self = shift;
    my $name = shift;
    my $value = shift;
    
    $self->get_exec_args->{$name} = $value;
}

=item B<process_args>

Process command line arguments.

=cut
sub process_args {
    my $self = shift;
    
    my $param;
    my @args;

    # arguments
    push(@args, @{$self->_load_options($self->get_params, $self->{_args})});
    push(@args, @{$self->_load_options($self->get_exec_params, $self->{_exec_args})});

    GetOptions(@args) || die("could not read command line arguments\n");
    
    if (@ARGV) {
        die("unrecognised options '" . join(' ', @ARGV) . "'\n");
    }
}

=item B<exec>

Execute program. Uses an C<exec()> call, so that if successful, never
returns.

=cut
sub exec {
    my $self = shift;

    my $builddir = $self->{_builddir};
    my @argv;
    
    my $key;
    foreach $key (keys %{$self->{_args}}) {
        if (defined $self->{_args}->{$key}) {
            if (length($key) == 1) {
                push(@argv, "-$key");
                push(@argv, $self->{_args}->{$key});
            } else {
                push(@argv, "--$key=" . $self->{_args}->{$key});
            }
        }
    }

    if ($self->get_named_exec_arg('with-cuda-gdb')) {
        unshift(@argv, "libtool --mode=execute cuda-gdb -q -ex run --args $builddir/" . $self->{_binary});        
    } elsif ($self->get_named_exec_arg('with-cuda-memcheck')) {
        unshift(@argv, "libtool --mode=execute cuda-memcheck $builddir/" . $self->{_binary});        
    } elsif ($self->get_named_exec_arg('with-gdb')) {
        unshift(@argv, "libtool --mode=execute gdb -q -ex run --args $builddir/" . $self->{_binary});
    } elsif ($self->get_named_exec_arg('with-valgrind')) {
        unshift(@argv, "libtool --mode=execute valgrind --leak-check=full $builddir/" . $self->{_binary});
    } elsif ($self->get_named_arg('with-mpi')) {
        my $np = '';
        if ($self->is_named_arg('mpi-np')) {
            $np .= " -np " . int($self->get_named_arg('mpi-np'));
        }
        if ($self->is_named_arg('mpi-npernode')) {
        	$np .= " -npernode " . int($self->get_named_arg('mpi-npernode'));
        }
        unshift(@argv, "mpirun$np $builddir/" . $self->{_binary});
    } else {
        unshift(@argv, "$builddir/" . $self->{_binary});
    }

    my $cmd = join(' ', @argv);
    if ($self->{_verbose}) {
        print "$cmd\n";
    }
    
    exec($cmd) || die("exec failed ($!)\n");
}

=item B<_load_options>(I<params>, I<args>)

Set up Getopt::Long specification using I<params> parameters, to write
arguments to I<args>. Return the specification.

=cut
sub _load_options {
	my $self = shift;
	my $params = shift;
	my $args = shift;
	
	my $param;
	my $arg;
	my @args;
	
	foreach $param (@$params) {
        if (defined $param->{default}) {
            $args->{$param->{name}} = $param->{default};
        }
        if ($param->{type} eq 'bool') {
        	my $name = $param->{name};
        	
	        push(@args, $name);
        	push(@args, sub { $args->{$param->{name}} = 1 });
        	
        	if ($name =~ s/^with\-/without\-/) {
        		push(@args, $name);
        		push(@args, sub { $args->{$param->{name}} = 0 });
        	} elsif ($name =~ s/^enable\-/disable\-/) {
        		push(@args, $name);
        		push(@args, sub { $args->{$param->{name}} = 0 });
        	}
        } else {
	        push(@args, $param->{name} . '=' . substr($param->{type}, 0, 1));
	        push(@args, \$args->{$param->{name}});
        }       
	}
	
	return \@args;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
