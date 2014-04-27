=head1 NAME

package - create and validate projects, build packages for distribution.

=head1 SYNOPSIS

    libbi package --create

    libbi package --validate

    libbi package --build
    
    libbi package --webify

=head1 DESCRIPTION

LibBi prescribes a standard structure for a project's model, configuration,
data and other files, and a standard format for packaging these.
The former assists with collaboration and reproducible research, the
latter with distribution.

The C<package> command provides facilities for creating a new LibBi project
with the standard file and directory structure, and bundling such a project
into a package for distribution.

=head2 A standard project

A standard project contains the following files:

=over 4

=item C<*.bi>

Model files.

=item C<init.sh>

A shell script that may be run to create any derived files. A common task is
to call C<libbi sample --target joint ...> to simulate a data set for testing
purposes.

=item C<run.sh>

A shell script that may be run to reproduce the results of the project.
Common tasks are to call C<libbi sample --target prior ...> to sample from the
prior distribution, and C<libbi sample --target posterior ...> to sample from
the posterior distribution. While a user may not necessarily run this script
directly, it should at least give them an idea of what the project is set up
to do, and what commands they might expect to work. After the C<README.md> file,
this would typically be the second file that a new user looks at.

=item C<*.conf>

Configuration files. Common files are C<prior.conf>, C<posterior.conf>,
C<filter.conf> and C<optimise.conf>, containing command-line options for
typical commands.

=item C<config.conf>

A particular configuration file where, by convention, a user can set any
platform-specific build and run options (such as C<--enable-cuda> and
C<--nthreads>). Any LibBi commands in the C<init.sh> and C<run.sh> scripts
should include this configuration file to bring in the user's own options
(e.g. C<libbi sample @config.conf ...>).

=item C<data/>

Directory containing data files that are passed to LibBi using the
C<--init-file>, C<--input-file> and C<--obs-file> command-line options.

=item C<results/>

Directory containing results files created from LibBi using the
C<--output-file> command-line option.

=back

=head2 A standard package

The following additional files are used for the packaging of a project:

=over 4

=item C<MANIFEST>

A list of files, one per line, to be included in the package.

=item C<LICENSE>

The license governing distribution of the package.

=item C<META.yml>

Meta data of the package. It should be formatted in YAML, giving the name,
author, version and description of the package. See that produced by
C<libbi package --create> as a guide.

=item C<README.md>

A description of the package. This would typically be the first file that a
new user looks at. It should be formatted in Markdown.

=item C<VERSION.md>

The version history of the package. It should be formatted in Markdown.

Packages should be given a three-figure version number of the form C<x.y.z>,
where C<x> is the version number, C<y> the major revision number and C<z>
the minor revision number. The version number would typically be incremented
after an overhaul of the package, the major revision number after the
addition of new functionality, and the minor revision number after bug fixes
or corrections. When a number is incremented, those numbers on the right
should be reset to zero. The first version number of a working package
should be C<1.0.0>. If a package is incomplete, only partially working or
being tested, the version number may be zero.

=back 

A project may contain any additional files, and these may be listed in the
C<MANIFEST> file for distribution. Commonly included are Octave, MATLAB
or R scripts for collating and plotting results, for example.

A standard package is simply a gzipped TAR archive with a file name of
C<I<Name>-I<Version>.tar.gz>. Extracting the archive produces a directory
with a name of C<I<Name>-I<Version>>, within which are all of those files
listed in the C<MANIFEST> file of the project.

=head2 Version control

Under version control, all project files with the exception of the following
would be included:

=over 4

=item * Any files in the C<results/> directory. These can be large, and at any
rate are derived files that a user should be able to reproduce for
themselves, perhaps with the C<run.sh> script.

=item * The C<results/> directory itself. This is always created automatically
when used in the C<--output-file> command-line option, and so its inclusion
is unnecessary. Moreover, it is common to
create a C<results> symlink to another directory where output files should
be written, particularly in a cluster environment where various network file
systems are available. Inclusion of the C<results/> directory in a version
control system becomes a nuisance in such cases.

=back

These files would also not typically be included in the package C<MANIFEST>.

=cut

package Bi::Client::package;

use parent 'Bi::Client', 'Bi::Gen';
use warnings;
use strict;

use Bi qw(share_file share_dir);
use Bi::Gen::Bi;

use Archive::Tar;
use File::Slurp;

=head1 OPTIONS

The following options are supported:

=over 4

=item C<--create>

Set up the current working directory as a LibBi project. This creates all the
standard files for a LibBi package with placeholder contents. It will prompt
to overwrite existing files.

=item C<--validate>

Validate the current working directory as a LibBi project.

=item C<--build>

Validate the current working directory as a LibBi project and build the
package. This produces a C<Model-x.y.z.tar.gz> file in the current working
directory for distribution.

=item C<--webify>

Create a file for publishing the package on a Jekyl website (such as
www.libbi.org). This produces a C<Model.md> file in the current working
directory.

=back

=head2 Package creation-specific options

=over 4

=item C<--name> (default 'Untitled')

Name of the package.

=back

=head1 METHODS

=over 4

=cut
our @CLIENT_OPTIONS = (
    {
      name => 'create',
      type => 'bool',
      default => 0
    },
    {
      name => 'validate',
      type => 'bool',
      default => 0
    },
    {
      name => 'build',
      type => 'bool',
      default => 0
    },
    {
      name => 'webify',
      type => 'bool',
      default => 0
    },
    {
      name => 'name',
      type => 'string',
      default => 'Untitled'
    },    
);

sub init {
    my $self = shift;

    $self->{_binary} = undef;
    push(@{$self->{_params}}, @CLIENT_OPTIONS);
    
    my $ttdir = share_dir(File::Spec->catdir('tt', 'package'));
    my $outdir = '.';
    
    %$self = (%$self, %{Bi::Gen->new($ttdir, $outdir)});    
}

sub is_cpp {
    return 0;
}

sub exec {
    my $self = shift;
    
    if ($self->get_named_arg('create')) {
        $self->create;
        $self->validate;
    }
    if ($self->get_named_arg('validate')) {
        $self->validate;
    }
    if ($self->get_named_arg('build')) {
        $self->validate;
        $self->build;
    }
    if ($self->get_named_arg('webify')) {
        $self->webify;
    }
}

sub needs_model {
    return 0;
}

=item C<create>

Create a standard project in the current working directory.

=cut
sub create {
    my $self = shift;

    # check inputs
    my $name = $self->get_named_arg('name');
    if ($name !~ /^[A-Z]/ || $name =~ /_/) {
        warn("Package name should be in CamelCase.\n");
    }
        
    # create files
    my $meta = {
        'name' => $name
    };
    my @files = (
        'MANIFEST',
        'LICENSE',
        'META.yml',
        'README.md',
        'VERSION.md',
        'run.sh',
        'init.sh',
        'config.conf'
    );    
    foreach my $file (@files) {
        if (!-e $file || $self->_yesno_prompt("$file already exists, overwrite?")) {
            $self->process_template("$file.tt", $meta, $file);
        }
    }
    if (!-e "$name.bi" || $self->_yesno_prompt("$name.bi already exists, overwrite?")) {
        $self->process_template("Model.bi.tt", $meta, "$name.bi");
    }
    
    # create directories
    mkdir('data');
}

=item C<validate>

Validate the current working directory as a standard project. This will print
warning messages for any issues detected.

=cut
sub validate {
    my $self = shift;
    
    my @manifest;
    my %manifest;
    my $file;
    my $contents;
    
    # check MANIFEST
    if (!-e 'MANIFEST') {
        warn("No MANIFEST. Create a MANIFEST file with a list of files, one per line, to be contained in the package.\n");
    } else {
        @manifest = $self->_read_manifest();
        map { $manifest{$_} = 1 } @manifest;
    }
    foreach $file (@manifest) {
        if (!-e $file) {
            warn("MANIFEST includes non-existent file $file.\n");
        }
    }

    # check LICENSE
    if (!-e 'LICENSE') {
        warn("No LICENSE. Create a LICENSE file containing the distribution license (e.g. GPL or BSD) of the package.\n");
    } elsif (!exists $manifest{'LICENSE'}) {
        warn("MANIFEST does not include LICENSE.\n");
    }
    
    # check META.yml
    if (!-e 'META.yml') {
        warn("No META.yml. Create a META.yml file containing the meta-data of the package in YAML format.\n");
    } else {
        if (!exists $manifest{'META.yml'}) {
            warn("MANIFEST does not include META.yml.\n");
        }
        my $meta = $self->_read_meta;
        if (exists $meta->{version} && $meta->{version} !~ /^\d+\.\d+\.\d+$/) {
            warn("Package version should be in the form x.y.z.\n");
        }
    }
    
    # check README.md
    if (!-e 'README.md') {
        warn("No README.md. Create a README.md file documenting the package in Markdown format.\n");
    } elsif (!exists $manifest{'README.md'}) {
        warn("MANIFEST does not include README.md.\n");
    }
    
    # check VERSION.md    
    if (!-e 'VERSION.md') {
        warn("No VERSION.md. Create a VERSION.md file to document changes to the package in Markdown format.\n");
    } elsif (!exists $manifest{'VERSION.md'}) {
        warn("MANIFEST does not include VERSION.md.\n");
    }

    # check run.sh
    if (!-e 'run.sh') {
        warn("No run.sh. Consider creating a run.sh script containing a few calls to LibBi for producing some meaningful experimental results from the package.\n");
    } else {
        if (!exists $manifest{'run.sh'}) {
            warn("MANIFEST does not include run.sh.\n");
        }
    }

    # check config.conf
    if (!-e 'config.conf') {
        warn("No config.conf. Consider creating a config.conf file where a user can put platform-specific configuration options.\n");
    }
    
    # check for things that might be missing from manifest
    my @maybe = (<*.bi>, <*.conf>, <*.sh>, <data/*.nc>, <oct/*.m>);
    foreach $file (@maybe) {
        if (!exists $manifest{$file}) {
            warn("Is $file missing from MANIFEST?\n");
        }
    }
}

=item C<build>

Build a distributable package from the current working directory.

=cut
sub build {
    my $self = shift;
    
    my $meta = $self->_read_meta;
    my @manifest = $self->_read_manifest;
    
    my $name = defined $meta->{name} ? $meta->{name} : 'Untitled';
    my $version = defined $meta->{version} ? $meta->{version} : '1.0.0';
    my $package = "$name-$version";
    
    my $tar = new Archive::Tar;
    foreach my $file (@manifest) {
        $tar->add_files($file);
        $tar->rename($file, File::Spec->catfile($package, $file));
    }    
    $tar->write("$package.tar.gz", COMPRESS_GZIP);
}

=item C<webify>

Create a markdown file for publishing as website with Jekyll.

=cut
sub webify {
    my $self = shift;
    
    my $name = $self->_read_meta->{name};
    my $file = "$name.md";

    my $meta = read_file('META.yml');
    my $readme = read_file('README.md');
    chomp $meta;
    chomp $readme;
    
    # remove first heading from README.md contents
    $readme =~ s/^(.*?)\n={3,}\n*//m;
    
    my $webpage = <<End;
---
layout: package
$meta
---

$readme
End

    if (!-e $file || $self->_yesno_prompt("$file already exists, overwrite?")) {
        write_file($file, $webpage);
    }
}

=item C<_read_manifest>()

Reads in the C<MANIFEST> file and returns a list of all files included.

=cut
sub _read_manifest {
    my $self = shift;
    
    open(MANIFEST, 'MANIFEST') || die("Could not open MANIFEST.\n");
    my @manifest = <MANIFEST>;
    close MANIFEST;
    chomp(@manifest);
    
    return @manifest;
}

=item C<_read_meta>()

Reads in the front-matter of the C<README.md> file and returns it as a
hashref.

=cut
sub _read_meta {
    my $self = shift;
    my $meta = {};
    my $line;
    my $key;
    my $lineno = 1;
    my $on = 0;

    # TODO: Use YAML or YAML::Tiny module
    open(META, 'META.yml') || die("Could not open META.yml\n");
    while ($line = <META>) {
        if ($line =~ /^(\w+): *(.*?)$/) {
            $key = lc($1);
            $meta->{$key} = $2;
        } elsif (defined $key) {
            $meta->{$key} .= $line;
        } else {
            warn("Line $lineno of META.yml is unrecognised.");
        }
        ++$lineno;
    }
    close META;    
    
    return $meta;
}

=item C<_yesno_prompt>(I<question>)

Prompt user for a yes or no answer.

=cut
sub _yesno_prompt {
    my $self = shift;
    my $question = shift;
    
    print "$question [y/N] ";
    my $answer = <>;
    chomp $answer;
    
    return $answer =~ /^y/i;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
