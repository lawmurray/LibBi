=head1 NAME

Bi::Client - generic client program.

=head1 SYNOPSIS

    use Bi::Client;
    my $client = new Bi::Client($cmd, $builddir);
    $client->process_args;
    $client->exec;  # if successful, never returns

=head1 RUN OPTIONS

Options that start with C<--with-> may be negated by instead starting them
with C<--without->.

=over 4

=item C<--dry-run>

Do not run.

=item C<--seed> (default automatic)

Pseudorandom number generator seed.

=item C<--nthreads I<N>> (default 0)

Run with C<N> threads. If zero, the number of threads used is the
default for OpenMP on the platform.

=item C<--with-gdb> (default off)

Run within the C<gdb> debugger.

=item C<--with-valgrind> (default off)

Run within C<valgrind>.

=item C<--with-cuda-gdb> (default off)

Run within the C<cuda-gdb> debugger.

=item C<--with-cuda-memcheck> (default off)

Run within C<cuda-memcheck>.

=item C<--gperftools-file> (default automatic)

Output file to use under C<--enable-gperftools>. The default is
C<I<command>.prof>.

=item C<--mpi-np>

Number of processes under C<--enable-mpi>, corresponding to the C<-np>
option to C<mpirun>.

=item C<--mpi-npernode>

Number of processes per node under C<--enable-mpi>. Corresponds to the
C<-npernode> option to C<mpirun>.

=item C<--mpi-hostfile>

Host file under C<--enable-mpi>, corresponding to the C<-hostfile> option to
C<mpirun>.

=item C<--role> (default C<client>)

When a client-server architecture is used under MPI, the role of the process;
either C<client> or C<server>.

=item C<--server-file> (default C<port_name>)

When a client-server architecture is used under MPI, the file containing
server connection information. A server process will write to this file, a
client process will read from it.

=back

=head1 COMMON OPTIONS

The options in this section are common across all client programs.

=head2 Input/output options

=over 4

=item C<--model-file>

File containing the model specification.

=item C<--init-file>

File from which to initialise parameters and initial conditions.

=item C<--input-file>

File from which to read inputs.

=item C<--obs-file>

File from which to read observations.

=item C<--output-file> (default automatic)

File to which to write output. The default is C<results/I<command>.nc>.

=item C<--init-ns> (default 0)

Index along the C<ns> dimension of C<--init-file> to use.

=item C<--init-np> (default -1)

Index along the C<np> dimension of C<--init-file> to use. -1 indicates that,
rather than initialising all state variables identically, the C<np> dimension
is used to initialise them all differently. The size of the C<np> dimension
must be at least the number of samples.

=item C<--input-ns> (default 0)

Index along the C<ns> dimension of C<--input-file> to use.

=item C<--input-np> (default -1)

Index along the C<np> dimension of C<--input-file> to use. -1 indicates that,
when state variables are read from the input file, rather than initialising all
state variables identically, the C<np> dimension is used to initialise them all
differently. The size of the C<np> dimension must be at least the number of
samples.

=item C<--obs-ns> (default 0)

Index along the C<ns> dimension of C<--obs-file> to use.

=item C<--obs-np> (default 0)

Index along the C<np> dimension of C<--obs-file> to use.

=back

=head2 Model transformations

=over 4

=item C<--with-transform-extended> (default automatic)

Transform the model for use with the extended Kalman filter. This includes
symbolically deriving Jacobian expressions.

=item C<--with-transform-param-to-state> (default automatic)

Treat parameters as state variables. This is useful for joint state and
parameter estimation using filters.

=item C<--with-transform-obs-to-state> (default automatic)

Treat observed variables as state variables. This is useful for producing
simulated data sets from a model.

=item C<--with-transform-initial-to-param> (default off)

Augment the parameters of the model with the initial values of state
variables. This is useful when sampling from the posterior distribution.
Treating initial values as parameters means that they are sampled using local
Metropolis-Hastings moves rather than by importance sampling, which may be
beneficial if a good importance proposal cannot be devised.

=back

=cut

package Bi::Client;

use warnings;
use strict;

use Carp::Assert;
use Cwd qw(abs_path);
use Getopt::Long qw(:config pass_through no_auto_abbrev no_ignore_case);
use File::Spec;
use File::Path qw(mkpath);

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
      name => 'init-file',
      type => 'string',
      default => ''
    },
    {
      name => 'input-file',
      type => 'string',
      default => ''
    },
    {
      name => 'obs-file',
      type => 'string',
      default => ''
    },
    {
      name => 'output-file',
      type => 'string',
      default => ''
    },
    {
      name => 'init-ns',
      type => 'int',
      default => 0
    },
    {
      name => 'init-np',
      type => 'int',
      default => -1
    },
    {
      name => 'input-ns',
      type => 'int',
      default => 0
    },
    {
      name => 'input-np',
      type => 'int',
      default => -1
    },
    {
      name => 'obs-ns',
      type => 'int',
      default => 0
    },
    {
      name => 'obs-np',
      type => 'int',
      default => 0
    },
    {
      name => 'seed',
      type => 'int',
      default => 0
    },
    {
      name => 'nthreads',
      type => 'int',
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
      # this is not usually set by users and is not documented, it is set
      # in Bi::Builder when --enable-mpi is used
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
      name => 'mpi-hostfile',
      type => 'string'
    },
    {
      name => 'role',
      type => 'string',
      default => 'client'
    },
    {
      name => 'server-file',
      type => 'string',
      default => 'port_name'
    },
    {
      name => 'with-transform-extended',
      type => 'bool',
      default => 0
    },
    {
      name => 'with-transform-param-to-state',
      type => 'bool'
    },
    {
      name => 'with-transform-obs-to-state',
      type => 'bool'
    },
    {
      name => 'with-transform-initial-to-param',
      type => 'bool',
      default => 0
    },
    {
      name => 'with-transform-iterated-filtering',
      type => 'bool',
      default => 0
    },
    {
      name => 'with-transform-iterated-smoothing',
      type => 'bool',
      default => 0
    },
    {
      name => 'with-transform-optimise',
      type => 'bool',
      default => 1
    },
    
    # deprecations
    {
      name => 'transform-extended',
      type => 'bool',
      deprecated => 1,
      message => "use --with-transform-extended instead"
    },
    {
      name => 'transform-param-to-state',
      type => 'bool',
      deprecated => 1,
      message => "use --with-transform-param-to-state instead"
    },
    {
      name => 'transform-obs-to-state',
      type => 'bool',
      deprecated => 1,
      message => "use --with-transform-obs-to-state instead"
    },
    {
      name => 'transform-initial-to-param',
      type => 'bool',
      deprecated => 1,
      message => "use --with-transform-initial-to-param instead"
    },
    {
      name => 'threads',
      type => 'int',
      deprecated => 1,
      message => 'use --nthreads instead'
    },
    {
      name => 'with-output',
      type => 'bool',
      deprecated => 1,
      message => 'omit --output-file, or use --output-file "" to disable output instead'
    },
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
    if ($cmd eq 'simulate') {
        die("the 'simulate' command is deprecated, use the 'sample' command and add '--target prior' instead\n");
    } elsif ($cmd !~ /^\w+$/) {
    	die("don't know what to do with command '$cmd'\n");
    }
    $class = "Bi::Client::$cmd";
    unless (eval("require $class")) {
        $class = "Bi::Test::$cmd";
        eval("require $class") || die("don't know what to do with command '$cmd'\n");
    }
    bless $self, $class;
        
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

=item B<needs_transform>

Does this client need the model to be transformed?

=cut
sub needs_transform {
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

=item B<delete_named_arg>(I<name>)

Delete the named argument of name I<name>.

=cut
sub delete_named_arg {
    my $self = shift;
    my $name = shift;
    
    delete $self->get_args->{$name};
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

=item B<is_named_exec_arg>(I<name>)

Is there a named argument of name I<name>?

=cut
sub is_named_exec_arg {
    my $self = shift;
    my $name = shift;
    
    return exists $self->get_exec_args->{$name} && defined $self->get_exec_args->{$name};
}

=item B<delete_named_exec_arg>(I<name>)

Delete the named argument of name I<name>.

=cut
sub delete_named_exec_arg {
    my $self = shift;
    my $name = shift;
    
    delete $self->get_exec_args->{$name};
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

    # check deprecated arguments
    foreach $param (@{$self->get_params}) {
        if ($param->{deprecated} && $self->is_named_arg($param->{name})) {
        	my $msg = "--" . $param->{name} . " is deprecated";
        	if (defined $param->{message}) {
        		$msg .= ", " . $param->{message};
        	}
        	$msg .= "\n";
        	warn($msg);
            $self->delete_named_arg($param->{name});
        }
    }
    if (@ARGV) {
        die("unrecognised options '" . join(' ', @ARGV) . "'\n");
    }
    
    # create output file directory if necessary
    my ($vol, $dir, $file) = File::Spec->splitpath($self->get_named_arg('output-file'));
    $dir = File::Spec->catpath($vol, $dir);
    mkpath($dir);
}

=item B<exec>

Execute program. Uses an C<exec()> call, so that if successful, never
returns.

=cut
sub exec {
    my $self = shift;

    my $builddir = $self->{_builddir};
    my $exeext = ($^O eq 'cygwin' || $^O eq 'MSWin32') ? '.exe' : '';
    my $binary = File::Spec->catfile($builddir, $self->get_binary . $exeext);
    my @argv;
    
    my $key;
    foreach $key (sort keys %{$self->{_args}}) {
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
        unshift(@argv, "cuda-gdb -q -ex run --args \"$binary\"");        
    } elsif ($self->get_named_exec_arg('with-cuda-memcheck')) {
        unshift(@argv, "cuda-memcheck \"$binary\"");
    } elsif ($self->get_named_exec_arg('with-gdb')) {
        unshift(@argv, "gdb -q -ex run --args \"$binary\"");
    } elsif ($self->get_named_exec_arg('with-valgrind')) {
        unshift(@argv, "valgrind --leak-check=full \"$binary\"");
    } else {
        unshift(@argv, "\"$binary\"");
    }
    if ($self->get_named_arg('with-mpi')) {
        my $np = '';
        if ($self->is_named_arg('mpi-np')) {
            $np .= " -np " . int($self->get_named_arg('mpi-np'));
        }
        if ($self->is_named_arg('mpi-npernode')) {
        	$np .= " -npernode " . int($self->get_named_arg('mpi-npernode'));
        }
        if ($self->is_named_arg('mpi-hostfile')) {
            $np .= " -hostfile " . $self->get_named_arg('mpi-hostfile');
        }
        unshift(@argv, "mpirun$np ");
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
	    if ($param->{name} eq 'seed') {
	        # special case for random number seed, putting this in the
	        # default field means that the C++ code for reading command-line
	        # options changes each call, forcing a recompile
	        $args->{$param->{name}} = int(rand(2**31 - 1));
	    } elsif (defined $param->{default}) {
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
