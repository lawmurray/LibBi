=head1 NAME

Bi::Gen - generic code generator.

=head1 SYNOPSIS

Use subclass.

=head1 REQUIRES

L<Template>

=head1 METHODS

=over 4

=cut

package Bi::Gen;

use warnings;
use strict;

use Carp::Assert;
use Template;
use Template::Filters;
use Template::Exception;
use IO::File;
use File::Compare;
use File::Copy;
use File::Path;
use File::Spec;
use File::Slurp;
use File::Find;

use Bi qw(share_file share_dir);
use Bi::Model;
use Bi::Expression;
use Bi::Visitor::ToCpp;

=item B<new>(I<ttdirs>, I<outdir>)

Constructor.

=over 4

=item I<ttdirs> 

Array reference, of directories containing Perl Template
Toolkit templates, or scalar giving single directory.

=item I<outdir> (optional)

Directory in which to output results.

=back

Returns the new object.

=cut
sub new {
    my $class = shift;
    my $ttdirs = shift;
    my $outdir = shift;

    if (ref($ttdirs) ne 'ARRAY') {
        $ttdirs = [ $ttdirs ];
    }
    my $tt = Template->new({
        INCLUDE_PATH => $ttdirs,
        FILTERS => {},
        RECURSION => 1,
        STRICT => 1
    }) || _error($Template::ERROR);
        
    my $self = {
        _tt => $tt,
        _ttdirs => $ttdirs,
        _outdir => $outdir
    };
    bless $self, $class;
    
    return $self;
}

=item B<get_tt>

Get Perl template toolkit object.

=cut
sub get_tt {
    my $self = shift;
    return $self->{_tt};
}

=item B<is_template>(I<filename>)

Does a template of the given I<filename> exist?

=cut
sub is_template {
    my $self = shift;
    my $filename = shift;
    
    my $ttdir;
    foreach $ttdir (@{$self->{_ttdirs}}) {
        if (-e "$ttdir/$filename") {
            return 1;
        }        
    }
    return 0;
}

=item B<process_template>(I<template>, I<vars>, I<output>)

Process template, preserving timestamp of output if unchanged.

=over 4

=item I<template> Template file name.

=item I<vars> Hash to be passed to template processor.

=item I<output> Output file name, or C<undef> for no output.

=back

No return value.

=cut
sub process_template {
    my $self = shift;
    my $template = shift;
    my $vars = shift;
    my $output = shift;
    
    my $null = undef;
    my $tt = $self->{_tt};
    my $to_file = defined($output) && ref($output) ne 'GLOB';
    my $out;
    if ($to_file) {
        $out = File::Spec->catfile($self->{_outdir}, $output);
    } elsif (!ref($output) eq 'GLOB') {
        $out = \$null;
    }
    my ($vol, $dir, $file);
    
    # create build directory
    mkpath($self->{_outdir});
    
    if ($to_file) {
        # write to temp file first, only copy over desired output if file
        # contents changes; keeps last modified dates, used by make,
        # consistent with actual file changes
        my $str;
        $tt->process($template, $vars, \$str) || _error($tt->error());
        if (!-e $out || read_file($out) ne $str) {
            ($vol, $dir, $file) = File::Spec->splitpath($out);
            mkpath($dir);
            write_file($out, $str);
        }
        if ($output =~ /\.sh$/) {
            chmod(0755, $out);
        }
    } else {
        $tt->process($template, $vars, $out) || _error($tt->error());
    }
}

=item B<copy_file>(I<src>, I<dst>)

Copy file, preserving timestamp of destination if unchanged.

=over 4

=item I<src> Source file name, relative to share directory.

=item I<dst> Destination file name, relative to output directory.

=back

No return value.

=cut
sub copy_file {
    my $self = shift;
    my $src = shift;
    my $dst = shift;
    
    my $in = share_file($src);
    my $out = File::Spec->catfile($self->{_outdir}, $dst);

    $self->_copy_file($in, $out);    
}

=item B<copy_dir>(I<src>, I<dst>, I<exts>)

Recursively copy directory, preserving timestamp of destination files if
unchanged.

=over 4

=item I<src> Source directory name, relative to share directory.

=item I<dst> Destination directory name, relative to output directory.

=item I<exts> Optional array ref of file extensions to include. If not given,
all files are included.

=back

No return value.

=cut
sub copy_dir {
    my $self = shift;
    my $src = shift;
    my $dst = shift;
    my $exts = shift;

    my $in = File::Spec->canonpath(share_dir($src));
    my $out = File::Spec->canonpath(File::Spec->catdir($self->{_outdir}, $dst));
    my $regexp;
    if (@$exts) {
        $regexp = join('|', @$exts);
        $regexp = qr/\.$regexp\$/;
    } else {
        $regexp = qr/./;
    }
    
    find({
        no_chdir => 1,
        wanted => sub {
            my $from = $File::Find::name;
            my $to = $from;
            $to =~ s/^$in/$out/i;
            my ($invol, $indir, $infile) = File::Spec->splitpath($in);
                        
            if (-f $from && $from =~ $regexp) {
                $self->_copy_file($from, $to);
            } elsif ($from =~ /\.svn$/) {
                $File::Find::prune = 1;
            }
        }
    }, $in);
}

=item B<_copy_file>(I<src>, I<dst>)

Copy file, preserving timestamp of destination if unchanged.

=over 4

=item I<src> Source file name, relative to working directory.

=item I<dst> Destination file name, relative to working directory.

=back

No return value.

=cut
sub _copy_file {
    my $self = shift;
    my $src = shift;
    my $dst = shift;
    
    my $in = File::Spec->catfile($src);
    my ($invol, $indir, $infile) = File::Spec->splitpath($in);

    my $out = File::Spec->catfile($dst);
    my ($outvol, $outdir, $outfile) = File::Spec->splitpath($out);
    
    # only copy over desired output if file contents changes; keeps last
    # modified dates, used by make, consistent with actual file changes
    my $str = read_file($in);
    if (!-e $out || read_file($out) ne $str) {
        mkpath($outdir);
        write_file($out, $str);
    }
    if ($out =~ /\.sh$/) {
        chmod(0755, $out);
    }
}

=back

=head1 CLASS METHODS

=item B<_error>(I<msg>)

Print I<msg> as error.

=cut
sub _error {
    my $msg = shift;

    # Perl Template Tookit puts exception name at start of error string,
    # remove this if present
    $msg =~ s/.*? - //;

    die("$msg\n");
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
