=head1 NAME

package - create and validate projects, build packages for distribution.

=head1 SYNOPSIS

    bi package --create

    bi package --validate

    bi package --build

=head1 DESCRIPTION

LibBi prescribes a standard structure for a project's model, configuration,
data and other files, and a standard format for packaging all of these. This
is useful for distribution, but is also motivated by the principle of
reproducible research: the standard format includes provision for executable
scripts that reproduce the results of a study.

The C<package> command provides facilities for creating a new LibBi project
with the standard file and directory structure, and bundling such a project
into a package for distribution.

=head2 A standard project

A standard project contains the following files:

=over 4

=item C<MANIFEST>

Contains a list of files, one per line, to be included when the project is
packaged for distribution.

=item C<README.txt>

A description of the package.

=item C<LICENSE.txt>

The license of the package.

=item C<VERSION.txt>

The version history of the package. Packages should be given a three-figure
version number of the form C<x.y.z> where C<x> is the version number, C<y>
the major revision number and C<z> the minor revision number. The version
number would typically be implemented after an overhaul of the package, the
major revision number after the addition of new functionality, and the minor
revision number after bug fixes or corrections. When a number is incremented,
those numbers on the right should be reset to zero. The first version number
of a working package should be C<1.0.0>. If a package is incomplete, only
partially working or being tested, the version number may be C<0.y.z>.

=item C<*.bi>

Model files, usually only one and given the same name as the package, i.e.
C<I<Name>.bi>, containing a model specification beginning
C<model I<Name> { ...>.

=item C<*.conf>

Configuration files, usually named after the client program with which they
are to be used, e.g. C<simulate.conf>, C<filter.conf> etc.

=item C<run.sh>

A shell script that may be run to reproduce the results of the project.

=item C<data/>

Contains data files that are passed to LibBi using the C<--init-file>,
C<--input-file> and C<--obs-file> command-line options.

=item C<results/>

Typically empty, but the C<--output-file> command-line option should be used
to write files to this directory.

=back

A project may contain any additional files, and these may be listed in the
C<MANIFEST> file for distribution. Commonly included are C<Octave>, C<MATLAB>
or C<R> scripts for collating and plotting results, for example.

=head2 A standard package

A standard package is simply a gzipped TAR archive with a file name of
C<I<Name>-I<Version>.tar.gz>. Extracting the archive produces a directory
with a name of C<I<Name>-I<Version>>, within which are all of those files
listed in the C<MANIFEST> file of the project.

=cut

package Bi::Client::package;

use base 'Bi::Client', 'Bi::Gen';
use warnings;
use strict;

use Bi qw(share_file share_dir);
use Bi::Gen::Bi;

use Archive::Tar;

=head1 OPTIONS

The following options are supported:

=over 4

=item C<--create>

Set up the current working directory as a LibBi project. This creates all the
standard files for a LibBi package with placeholder contents.

=item C<--validate>

Validate the current working directory as a LibBi project.

=item C<--build>

Validate and build the current working directory as a LibBi project. This
produces a C<Model-x.y.z.tar.gz> file in the current working directory for
distribution.

=back

=head2 Package creation-specific options

=over 4

=item C<--name> (default 'Untitled')

Name of the package.

=item C<--version> (default '1.0.0')

Version number of the package.

=item C<--description> (default '')

One sentence description of the package.

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
      name => 'name',
      type => 'string',
      default => 'Untitled'
    },
    {
      name => 'version',
      type => 'string',
      default => '1.0.0'
    },
    {
      name => 'description',
      type => 'string',
      default => ''
    }
    
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
    
    my $version = $self->get_named_arg('version');
    if ($version !~ /^\d+\.\d+\.\d+$/) {
        warn("Package version should be in the form x.y.z.\n");
    }
    
    my $description = $self->get_named_arg('description');
    
    # create files
    my $meta = {
        'name' => $name,
        'version' => $version,
        'description' => $description
    };
    my @files = (
        'MANIFEST',
        'META',
        'README.txt',
        'LICENSE.txt',
        'VERSION.txt',
        'run.sh'
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
    mkdir('results');
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

    # check META
    if (!-e 'META') {
        warn("No META. Create a META file containing the name, version and a description of the package for automated use. It should have the following format:\nName: Name of the package\nVersion: Version number of the package\nDescription: Description of the package.\n");
    } else {
        if (!exists $manifest{'META'}) {
            warn("MANIFEST does not include META.\n");
        }
        $self->_read_meta; # to check format
    }

    # check README.txt
    if (!-e 'README.txt') {
        warn("No README.txt. Create a README.txt file documenting the package in a human-readable form.\n");
    } elsif (!exists $manifest{'README.txt'}) {
        warn("MANIFEST does not include README.txt.\n");
    }
    
    # check LICENSE.txt
    if (!-e 'LICENSE.txt') {
        warn("No LICENSE.txt. Create a LICENSE.txt file containing the distribution license (e.g. GPL or BSD) of the package.\n");
    } elsif (!exists $manifest{'LICENSE.txt'}) {
        warn("MANIFEST does not include LICENSE.txt.\n");
    }

    # check VERSION.txt    
    if (!-e 'VERSION.txt') {
        warn("No VERSION.txt. Create a VERSION.txt file to document changes to the package in a human-readable form.\n");
    } elsif (!exists $manifest{'VERSION.txt'}) {
        warn("MANIFEST does not include VERSION.txt.\n");
    }
    
    # check run.sh
    if (!-e 'run.sh') {
        warn("No run.sh. Consider creating a run.sh script containing a few calls to LibBi for producing some meaningful experimental results from the package.\n");
    } else {
        if (!exists $manifest{'run.sh'}) {
            warn("MANIFEST does not include run.sh.\n");
        }
    }
    
    # check for things that might be missing from manifest
    my @maybe = (<*.bi>, <*.conf>);
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

Reads in the C<META> files and returns it as a hashref.

=cut
sub _read_meta {
    my $self = shift;
    my $meta = {};
    my $line;
    my $key;
    my $lineno = 1;

    open(META, 'META') || die("Could not open META.\n");
    while ($line = <META>) {
        if ($line =~ /^(\w+): *(.*?)$/) {
            $key = lc($1);
            $meta->{$key} = $2;
        } elsif (defined $key) {
            $meta->{$key} .= $line;
        } else {
            warn("Line $lineno of META is unrecognised.");
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
