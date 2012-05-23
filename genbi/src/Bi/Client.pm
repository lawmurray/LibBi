=head1 NAME

Bi::Client - generic client program.

=head1 SYNOPSIS

    use Bi::Client
    my $client = new Bi::Client($dir);
    $client->process_args;
    $client->exec;  # if successful, never returns

=cut

package Bi::Client;

use warnings;
use strict;

use Carp::Assert;
use Cwd qw(abs_path);
use Getopt::Long qw(:config pass_through no_auto_abbrev);
use FindBin qw($Bin);

=head1 METHODS

=over 4

=item B<new>(I<cmd>, I<builddir>, I<verbose>, I<debug>)

Constructor.

=over 4

=item * I<cmd> The command.

=item * I<builddir> The build directory

=item * I<verbose> Is verbosity enabled?

=item * I<debug> Is debugging enabled?

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
    my $debug = shift;
    
    $builddir = abs_path($builddir);
    my $self = {
        _builddir => $builddir,
        _verbose => $verbose,
        _debug => $debug,
        _binary => '',
        _gdb => 0,
        _cuda_gdb => 0,
        _valgrind => 0,
        _params => [],
        _args => {}
    };
    
    # look up appropriate class name
    $class = "Bi::Client::$cmd";
    eval ("require $class") || die("don't know what to do with command '$cmd'\n");
    
    bless $self, $class;
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

=item B<get_params>

Get formal parameters of the client program as a hash. 

=cut
sub get_params {
    my $self = shift;
    return $self->{_params};
}

=item B<get_args>

Get arguments to the client program as a hash. These are available after a
call to B<process_args>.

=cut
sub get_args {
    my $self = shift;
    
    return $self->{_args};
}

=item B<is_named_arg>(I<name>)

Is there a named argument of name I<name>?

=cut
sub is_named_arg {
    my $self = shift;
    my $name = shift;
    
    return exists $self->get_args->{$name} && defined $self->get_args->{$name};
}

=item B<get_named_arg>(I<name>)

Get named argument to the client program.

=cut
sub get_named_arg {
    my $self = shift;
    my $name = shift;
    
    return $self->get_args->{$name};
}

=item B<get_linearise>

Should model be linearised for this client?

=cut
sub get_linearise {
    return 0;
}

=item B<process_args>

Process command line arguments.

=cut
sub process_args {
    my $self = shift;
    
    my $param;
    my @args = (
        'gdb!' => \$self->{_gdb},
        'cuda-gdb!' => \$self->{_cuda_gdb},
        'valgrind!' => \$self->{_valgrind}
    );

    # run arguments
    foreach $param (@{$self->{_params}}) {
        if (defined $param->{default}) {
            $self->{_args}->{$param->{name}} = $param->{default};
        }        
        push(@args, $param->{name} . '=' . substr($param->{type}, 0, 1));       
        push(@args, \$self->{_args}->{$param->{name}});
    }

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

    if ($self->{_cuda_gdb}) {
        unshift(@argv, "cuda-gdb -q -ex run --args $builddir/" . $self->{_binary});        
    } elsif ($self->{_gdb}) {
        unshift(@argv, "gdb -q -ex run --args $builddir/" . $self->{_binary});
    } elsif ($self->{_valgrind}) {
        unshift(@argv, "valgrind --leak-check=full $builddir/" . $self->{_binary});
    } else {
        unshift(@argv, "$builddir/" . $self->{_binary});
    }

    my $cmd = join(' ', @argv);
    if ($self->{_verbose}) {
        print "$cmd\n";
    }
    
    $ENV{LD_LIBRARY_PATH} .= ':' . File::Spec->catdir($Bin, '..', '..', 'lib', '.libs');
    exec($cmd) || die("exec failed ($!)\n");
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
