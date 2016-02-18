=head1 NAME

Bi::Builder - builds client programs on demand.

=head1 SYNOPSIS

    use Bi::Builder;
    my $builder = new Bi::Builder($dir);
    my $client = 'simulate';  # or 'filter' or 'smooth' etc
    $builder->build($client);

=head1 BUILD OPTIONS

Options that start with C<--enable-> may be negated by instead starting them
with C<--disable->.

=over 4

=item C<--dry-parse> (default off)

Do not parse model file. Implies C<--dry-gen>.

=item C<--dry-gen> (default off)

Do not generate code.

=item C<--dry-build> (default off)

Do not build.

=item C<--force> (default off)

Force all build steps to be performed, even when determined not to be
required.

=item C<--enable-warnings> (default off)

Enable compiler warnings.

=item C<--enable-assert> (default on)

Enable assertion checking. This is recommended for test runs, but not for
production runs.

=item C<--enable-extra-debug> (default off)

Enable extra debugging options in compilation. This is recommended along with
C<--with-gdb> or C<--with-cuda-gdb> when debugging.

=item C<--enable-diagnostics n> (default 0)

Enable diagnostic output n to standard error.

=item C<--enable-diagnostics2> (default off)

Enable another type of diagnostic.

=item C<--enable-single> (default off)

Use single-precision floating point.

=item C<--enable-openmp> (default on)

Use OpenMP multithreading.

=item C<--enable-cuda> (default off)

Enable CUDA code for graphics processing units (GPU).

=item C<--enable-gpu-cache> (default off)

For particle filters, enable ancestry caching in GPU memory. GPU memory is
typically much more limited than main memory. If sufficient GPU memory is
available this may give some performance improvement.

=item C<--enable-sse> (default off)

Enable SSE code.

=item C<--enable-avx> (default off)

Enable AVX code.

=item C<--enable-mpi> (default off)

Enable MPI code.

=item C<--enable-vampir> (default off)

Enable Vampir profiling.

=item C<--enable-gperftools> (default off)

Enable C<gperftools> profiling.

=back

=head1 METHODS

=over 4

=cut

package Bi::Builder;

use warnings;
use strict;

use Carp::Assert;
use Cwd qw(abs_path getcwd);
use Getopt::Long qw(:config pass_through no_auto_abbrev no_ignore_case);
use File::Spec;
use File::Slurp;
use File::Path;

=item B<new>(I<name>, I<verbose>)

Constructor.

=over 4

=item I<name> The model name.

=item I<verbose> Is verbosity enabled?

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $name = shift;
    my $verbose = int(shift);
    
    my $self = {
        _builddir => '',
        _verbose => $verbose,
        _force => 0,
        _warnings => 0,
        _assert => 1,
        _openmp => 1,
        _cuda => 0,
        _gpu_cache => 0,
        _sse => 0,
        _avx => 0,
        _mpi => 0,
        _vampir => 0,
        _single => 0,
        _extra_debug => 0,
        _timing => 0,
        _diagnostics => 0,
        _diagnostics2 => 0,
        _gperftools => 0,
        _tstamps => {},
    };
    bless $self, $class;
    
    # command line options
    my @args = (
        'force' => \$self->{_force},
        'enable-warnings' => sub { $self->{_warnings} = 1 },
        'disable-warnings' => sub { $self->{_warnings} = 0 },
        'enable-assert' => sub { $self->{_assert} = 1 },
        'disable-assert' => sub { $self->{_assert} = 0 },
        'enable-openmp' => sub { $self->{_openmp} = 1 },
        'disable-openmp' => sub { $self->{_openmp} = 0 },
        'enable-cuda' => sub { $self->{_cuda} = 1 },
        'disable-cuda' => sub { $self->{_cuda} = 0 },
        'enable-gpu-cache' => sub { $self->{_gpu_cache} = 1 },
        'disable-gpu-cache' => sub { $self->{_gpu_cache} = 0 },
        'enable-sse' => sub { $self->{_sse} = 1 },
        'disable-sse' => sub { $self->{_sse} = 0 },
        'enable-avx' => sub { $self->{_avx} = 1 },
        'disable-avx' => sub { $self->{_avx} = 0 },
        'enable-mpi' => sub { $self->{_mpi} = 1 },
        'disable-mpi' => sub { $self->{_mpi} = 0 },
        'enable-vampir' => sub { $self->{_vampir} = 1 },
        'disable-vampir' => sub { $self->{_vampir} = 0 },
        'enable-single' => sub { $self->{_single} = 1 },
        'disable-single' => sub { $self->{_single} = 0 },
        'enable-extra-debug' => sub { $self->{_extra_debug} = 1 },
        'disable-extra-debug' => sub { $self->{_extra_debug} = 0 },
        'enable-timing' => sub { $self->{_timing} = 1 },
        'disable-timing' => sub { $self->{_timing} = 0 },
        'enable-diagnostics=i' => \$self->{_diagnostics},
        'disable-diagnostics' => sub { $self->{_diagnostics} = 0 },
        'disable-diagnostics2' => sub { $self->{_diagnostics2} = 0 },
        'enable-gperftools' => sub { $self->{_gperftools} = 1 },
        'disable-gperftools' => sub { $self->{_gperftools} = 0 },
    );
    GetOptions(@args) || die("could not read command line arguments\n");
    
    # can't support AVX or SSE when CUDA enabled at this stage
    if ($self->{_cuda} && $self->{_avx}) {
    	warn("AVX has been disabled, unsupported when CUDA also enabled\n");
    	$self->{_avx} = 0;
    }
    if ($self->{_cuda} && $self->{_sse}) {
    	warn("SSE has been disabled, unsupported when CUDA also enabled\n");
    	$self->{_sse} = 0;
    }
    
    # some AVX instructions defer to SSE, so enable SSE too
    if ($self->{_avx}) {
    	$self->{_sse} = 1;
    }
    
    # enable mpirun automatically when --enable-mpi used
    if ($self->{_mpi}) {
        push(@ARGV, '--with-mpi');
    }

    # work out name of build directory
    my @builddir = 'build';
    push(@builddir, 'assert') if $self->{_assert};
    push(@builddir, 'openmp') if $self->{_openmp};
    push(@builddir, 'cuda') if $self->{_cuda};
    push(@builddir, 'gpucache') if $self->{_gpu_cache};
    push(@builddir, 'sse') if $self->{_sse};
    push(@builddir, 'avx') if $self->{_avx};
    push(@builddir, 'mpi') if $self->{_mpi};
    push(@builddir, 'vampir') if $self->{_vampir};
    push(@builddir, 'single') if $self->{_single};
    push(@builddir, 'extradebug') if $self->{_extra_debug};
    push(@builddir, 'diagnostics' . $self->{_diagnostics}) if $self->{_diagnostics};
    push(@builddir, 'gperftools') if $self->{_gperftools};
    
    $self->{_builddir} = File::Spec->catdir(".$name", join('_', @builddir));
    mkpath($self->{_builddir});
    
    # time stamp dependencies
    $self->_stamp(File::Spec->catfile($self->{_builddir}, 'autogen.sh'));
    $self->_stamp(File::Spec->catfile($self->{_builddir}, 'configure.ac'));
    $self->_stamp(File::Spec->catfile($self->{_builddir}, 'Makefile.am'));
    $self->_stamp(File::Spec->catfile($self->{_builddir}, 'configure'));
    $self->_stamp(File::Spec->catfile($self->{_builddir}, 'Makefile'));
    $self->_stamp(File::Spec->catfile($self->{_builddir}, 'bi.lpp'));
    $self->_stamp(File::Spec->catfile($self->{_builddir}, 'bi.ypp'));
    
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

=item B<get_dir>

Get the name of the build directory.

=cut
sub get_dir {
    my $self = shift;
    return $self->{_builddir};
}

=item B<_autogen>

Run the C<autogen.sh> script if one or more of it, C<configure.ac> or
C<Makefile.am> has been modified since the last run, or if the C<configure>
script does not exist (meaning its last execution failed).

No return value.

=cut
sub _autogen {
    my $self = shift;
    my $builddir = $self->get_dir;
        
    if ($self->{_force} ||
        $self->_is_modified(File::Spec->catfile($builddir, 'autogen.sh')) ||
        $self->_is_modified(File::Spec->catfile($builddir, 'configure.ac')) ||
        $self->_is_modified(File::Spec->catfile($builddir, 'Makefile.am')) ||
        !-e File::Spec->catfile($builddir, 'configure') ||
        !-e File::Spec->catfile($builddir, 'install-sh')) {
        my $cwd = getcwd();
        
        my $options = '';
        if (!$self->{_verbose}) {
            $options .= ' > autogen.log 2>&1'; 
        }
        
        chdir($builddir) || die("could not change to build directory '$builddir'\n");
        my $ret = system("./autogen.sh$options");
        if ($? == -1) {
            die("./autogen.sh failed to execute ($!)\n");
        } elsif ($? & 127) {
            die(sprintf("./autogen.sh died with signal %d. See $builddir/autogen.log for details\n", $? & 127));
        } elsif ($ret != 0) {
            die(sprintf("./autogen.sh failed with return code %d. Make sure autoconf and automake are installed. See $builddir/autogen.log for details\n", $ret >> 8));
        }
        chdir($cwd) || warn("could not change back to working directory '$cwd'\n");
    }
}

=item B<_configure>

Run the C<configure> script if it has been modified since the last run, or
if C<Makefile> does not exist (meaning its last execution failed).

No return value.

=cut
sub _configure {
    my $self = shift;

    my $builddir = $self->get_dir;
    my $cwd = getcwd();
    my $cxxflags = '-O3 -g3 -funroll-loops';
    my $linkflags = '';
    my $options = '';

    #if (!$self->{_force}) {
    #    $options .= ' --config-cache';
    #}

    $options .= $self->{_assert} ? ' --enable-assert' : ' --disable-assert';
    $options .= $self->{_openmp} ? ' --enable-openmp' : ' --disable-openmp';
    $options .= $self->{_cuda} ? ' --enable-cuda' : ' --disable-cuda';
    $options .= $self->{_gpu_cache} ? ' --enable-gpucache' : ' --disable-gpucache';
    $options .= $self->{_sse} ? ' --enable-sse' : ' --disable-sse';
    $options .= $self->{_avx} ? ' --enable-avx' : ' --disable-avx';
    $options .= $self->{_mpi} ? ' --enable-mpi' : ' --disable-mpi';
    $options .= $self->{_vampir} ? ' --enable-vampir' : ' --disable-vampir';
    $options .= $self->{_single} ? ' --enable-single' : ' --disable-single';
    $options .= $self->{_extra_debug} ? ' --enable-extradebug' : ' --disable-extradebug';
    $options .= $self->{_timing} ? ' --enable-timing' : ' --disable-timing';
    $options .= $self->{_diagnostics} ? ' --enable-diagnostics=' . $self->{_diagnostics} : ' --disable-diagnostics';
    $options .= $self->{_gperftools} ? ' --enable-gperftools' : ' --disable-gperftools';
    
    if ($self->{_extra_debug}) {
    	$cxxflags = '-O0 -g3 -fno-inline -D_GLIBCXX_DEBUG';
    }

    if (!$self->{_verbose}) {
        $options .= ' > configure.log 2>&1'; 
    }
            
    if ($self->{_warnings}) {
        $cxxflags .= " -Wall";
        $linkflags .= " -Wall";
    }

    if ($self->{_force} ||
        $self->_is_modified(File::Spec->catfile($builddir, 'configure')) ||
        !-e File::Spec->catfile($builddir, 'Makefile')) {
        
        my $cmd = "./configure $options CXXFLAGS='$cxxflags' LINKFLAGS='$linkflags'";
        if ($self->{_verbose}) {
            print "$cmd\n";
        }
        
        chdir($builddir);
        my $ret = system($cmd);
        if ($? == -1) {
            die("./configure failed to execute ($!)\n");
        } elsif ($? & 127) {
            die(sprintf("./configure died with signal %d. See $builddir/configure.log and $builddir/config.log for details\n", $? & 127));
        } elsif ($ret != 0) {
            die(sprintf("./configure failed with return code %d." . _configure_whats_missing('configure.log') . " See $builddir/configure.log and $builddir/config.log for details\n", $ret >> 8));
        }        
        chdir($cwd);
    }
}

=item B<_make>(I<client>)

Run C<make> to compile the given client program.

=over 4

=item I<client>

The name of the client program.

=back

No return value.

=cut
sub _make {
    my $self = shift;
    my $client = shift;
    
    my $exeext = ($^O eq 'cygwin' || $^O eq 'MSWin32') ? '.exe' : '';
    my $target = $client . "_" . ($self->{_cuda} ? 'gpu' : 'cpu') . $exeext;
    my $link = $client . $exeext;
    my $options = '';
    if ($self->{_force}) {
        $options .= ' --always-make';
    }
    if (!$self->{_verbose}) {
        $options .= ' > make.log 2>&1'; 
    }
    
    my $builddir = $self->get_dir;
    my $cwd = getcwd();
    my $cmd = "make -j 4 $options $target";
    
    if ($self->{_verbose}) {
        print "$cmd\n";
    }
    
    chdir($builddir);
    my $ret = system($cmd);
    if ($? == -1) {
        die("make failed to execute ($!)\n");
    } elsif ($? & 127) {
        die(sprintf("make died with signal %d, see $builddir/make.log for details\n", $? & 127));
    } elsif ($ret != 0) {
        die(sprintf("make failed with return code %d, see $builddir/make.log for details\n", $ret >> 8));
    }
    symlink($target, $link);
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

=item B<_configure_whats_missing>(I<configure_log>)

Try to work out what might be missing when configure fails. Return an
appropriate message.

=cut
sub _configure_whats_missing {
    my $configure_log = shift;
    
    my @lines = read_file($configure_log);
    my $helpful = '';
    if ($lines[$#lines] =~ /^configure: error: (.*?)$/) {
    	$helpful = " $1.";
    }
    return $helpful;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
