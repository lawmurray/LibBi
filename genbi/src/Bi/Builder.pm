=head1 NAME

Bi::Builder - builds client programs on demand.

=head1 SYNOPSIS

    use Bi::Builder
    my $builder = new Bi::Builder($dir);
    my $client = 'simulate';  # or 'filter' or 'smooth' etc
    $builder->build($client);

=head1 COMMAND LINE

The following arguments are read from the command line to control various
aspects of the build. Each may be prefixed with C<no-> to instead disable
the associated functionality.

=over 4

=item * C<--verbose>

Enable verbose reporting.

=item * C<--warn>

Enable compiler warnings.

=item * C<--debug>

Enable debugging mode. This will enable assertion checking.

=item * C<--profile>

Enable profiling with C<gprof>.

=item * C<--native>

Compile for native platform.

=item * C<--gpu>

Enable CUDA device code.

=item * C<--sse>

Enable SSE host code.

=item * C<--double>

Use double-precision floating point.

=item * C<--force>

Force all build steps to be performed, even when determined to be not
required. This is useful as a workaround of any faults in detecting
changes, or when system headers or libraries change and recompilation or
relinking is required.

=back

=head1 METHODS

=over 4

=cut

package Bi::Builder;

use warnings;
use strict;

use Carp::Assert;
use Cwd qw(abs_path getcwd);
use Getopt::Long qw(:config pass_through);
use File::Spec;

=item B<new>(I<builddir>, I<verbose>, I<debug>)

Constructor.

=over 4

=item * I<builddir> The build directory

=item * I<verbose> Is verbosity enabled?

=item * I<debug> Is debugging enabled?

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $builddir = shift;
    my $verbose = shift;
    my $debug = shift;
    
    $builddir = abs_path($builddir);
    my $self = {
        _builddir => $builddir,
        _verbose => $verbose,
        _debug => $debug,
        _warn => 0,
        _gpu => 0,
        _profile => 0,
        _native => 0,
        _sse => 0,
        _double => 1,
        _force => 0,
        _tstamps => {}
    };
    bless $self, $class;
    
    # command line options
    my @args = (
        'warn!' => \$self->{_warn},
        'profile!' => \$self->{_profile},
        'native!' => \$self->{_native},
        'gpu!' => \$self->{_gpu},
        'sse!' => \$self->{_sse},
        'double!' => \$self->{_double},
        'force!' => \$self->{_force}
    );
    GetOptions(@args) || die("could not read command line arguments\n");
    
    # time stamp dependencies
    $self->_stamp(File::Spec->catfile($builddir, 'autogen.sh'));
    $self->_stamp(File::Spec->catfile($builddir, 'configure.ac'));
    $self->_stamp(File::Spec->catfile($builddir, 'Makefile.am'));
    $self->_stamp(File::Spec->catfile($builddir, 'configure'));
    $self->_stamp(File::Spec->catfile($builddir, 'Makefile'));
    
    return $self;
}

=item B<build>(I<client>)

Build a client program.

=over 4

=item I<client> The name of the client program ('simulate', 'filter',
'pmcmc', etc).

=back

No return value.

=cut
sub build {
    my $self = shift;
    my $client = shift;
    
    $self->_autogen;
    $self->_configure;
    $self->_make($client);
}

=item B<_autogen>

Run the C<autogen.sh> script if one or more of it, C<configure.ac> or
C<Makefile.am> has been modified since the last run.

No return value.

=cut
sub _autogen {
    my $self = shift;
    my $builddir = $self->{_builddir};
        
    if ($self->{_force} ||
        $self->_is_modified(File::Spec->catfile($builddir, 'autogen.sh')) ||
        $self->_is_modified(File::Spec->catfile($builddir, 'configure.ac')) ||
        $self->_is_modified(File::Spec->catfile($builddir, 'Makefile.am'))) {          
        my $cwd = getcwd();
        
        chdir($builddir) || die("could not change to build directory '$builddir'\n");
        my $ret = system('./autogen.sh');
        if ($? == -1) {
            die("./autogen.sh failed to execute ($!)\n");
        } elsif ($? & 127) {
            die(sprintf("./autogen.sh died with signal %d\n", $? & 127));
        } elsif ($ret != 0) {
            die(sprintf("./autogen.sh failed with return code %d\n", $ret >> 8));
        }
        
        chdir($cwd) || warn("could not change back to working directory '$cwd'\n");
    }
}

=item B<_configure>

Run the C<configure> script if it has been modified since the last run.

No return value.

=cut
sub _configure {
    my $self = shift;

    my $builddir = $self->{_builddir};
    if ($self->{_force} ||
        $self->_is_modified(File::Spec->catfile($builddir, 'configure'))) {
        my $cwd = getcwd();
        my $cxxflags = '-O3 -funroll-loops';
        my $linkflags = '';
        my $options = '';
        
        if ($self->{_warn}) {
            $cxxflags .= " -Wall";
            $linkflags .= " -Wall";
        }
        if ($self->{_debug}) {
            $cxxflags .= ' -g3';
        } else {
            $cxxflags .= ' -g0';
            $options .= " --disable-assert";
        }
        if ($self->{_profile}) {
            $cxxflags .= " -pg";
            $linkflags .= " -pg";
        }
        if ($self->{_native}) {
            $cxxflags .= " -march=native";
        }
        if (!$self->{_debug} && !$self->{_profile}) {
            $cxxflags .= " -fomit-frame-pointer";
        }
        if (!$self->{_verbose}) {
            $options .= ' --silent'; 
        }
        $options .= $self->{_gpu} ? ' --enable-gpu' : ' --disable-gpu';
        $options .= $self->{_sse} ? ' --enable-sse' : ' --disable-sse';
        $options .= $self->{_double} ? ' --enable-double' : ' --disable-double';
        if (!$self->{_force}) {
            $options .= ' --config-cache';
        }
        
        chdir($builddir);
        my $ret = system("./configure $options CXXFLAGS='$cxxflags' LINKFLAGS='$linkflags'");
        if ($? == -1) {
            die("./configure failed to execute ($!)\n");
        } elsif ($? & 127) {
            die(sprintf("./configure died with signal %d\n", $? & 127));
        } elsif ($ret != 0) {
            die(sprintf("./configure failed with return code %d\n", $ret >> 8));
        }        
        chdir($cwd);
    }
}

=item B<_make>(I<client>)

Run C<make> to compile the given client program.

=over 4

=item I<client> The name of the client program ('simulate', 'filter',
'pmcmc', etc).

=back

No return value.

=cut
sub _make {
    my $self = shift;
    my $client = shift;
    
    my $target = $client . "_" . ($self->{_gpu} ? 'gpu' : 'cpu');
    my $options = '';
    if (!$self->{_verbose}) {
        $options .= ' --quiet'; 
    }
    if ($self->{_force}) {
        $options .= ' --always-make';
    }
    
    my $builddir = $self->{_builddir};
    my $cwd = getcwd();
    my $cmd = "make -j 4 $options $target";
    
    chdir($builddir);
    my $ret = system($cmd);
    if ($? == -1) {
        die("make failed to execute ($!)\n");
    } elsif ($? & 127) {
        die(sprintf("make died with signal %d\n", $? & 127));
    } elsif ($ret != 0) {
        die(sprintf("make failed with return code %d\n", $ret >> 8));
    }
    symlink($target, $client);
    chdir($cwd);
}

=item B<_stamp>(I<filename>)

Update timestamp on file.

=over 4

=item I<filename> The name of the file.

=back

No return value.

=cut
sub _stamp {
    my $self = shift;
    my $filename = shift;

    $self->{_tstamps}->{$filename} = _last_modified($filename);
}

=item B<_is_modified>(I<filename>)

Has file been modified since last use?

=over 4

=item I<filename> The name of the file.

=back

Returns true if the file has been modified since last use.

=cut
sub _is_modified {
    my $self = shift;
    my $filename = shift;

    return (!exists $self->{_tstamps}->{$filename} ||
        _last_modified($filename) < $self->{_tstamps}->{$filename});
}

=item B<_last_modified>(I<filename>)

Get time of last modification of a file.

=over 4

=item I<filename> The name of the file.

=back

Returns the time of last modification of the file, or the current time if
the file does not exist.

=cut
sub _last_modified {
    my $filename = shift;
    
    if (-e $filename) {
        return abs(-M $filename); # modified time in stat() has insufficient res
    } else {
        return time;
    }
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
