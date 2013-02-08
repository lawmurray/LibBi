=head1 NAME

LibBi - Bayesian inference for state-space models, including sequential Monte
Carlo (SMC), particle Markov chain Monte Carlo (PMCMC) and SMC^2 methods on
multithreaded, graphics processing unit (GPU) and distributed memory
architectures.

=head1 SYNOPSIS

    use Bi qw(share_file share_dir);

=head1 METHODS

=over 4

=cut

package Bi;

use base 'Exporter';
use warnings;
use strict;

our @EXPORT_OK = qw(share_file share_dir);

use FindBin qw($Bin);
use File::ShareDir qw(dist_file dist_dir);
use File::Spec;

=item B<share_file>(I<file>)

Returns a full path from which the shared file (not directory) I<file> can be
retrieved. A check for the existence of the file is made during the process.

=cut
sub share_file {
    my $file = shift;
    
    my $share_file = File::Spec->catfile($Bin, '..', 'share', $file);
    if (!-e $share_file || !-f $share_file) {
        $share_file = dist_file('Bi', $file);
        if (!-e $share_file || !-f $share_file) {
            die("could not find shared file $file\n");
        }
    }
    return $share_file;
}

=item B<share_dir>(I<dir>)

Returns a full path from which the shared directory I<dir> can be
retrieved. A check for the existence of the directory is made during the
process.

=cut
sub share_dir {
    my $dir = shift;
    
    my $share_dir = File::Spec->catdir($Bin, '..', 'share', $dir);
    if (!-e $share_dir || !-d $share_dir) {
        $share_dir = File::Spec->catdir(dist_dir('Bi'), $dir);
        if (!-e $share_dir || !-d $share_dir) {
            die('could not find shared directory $dir\n')
        }
    }
    return $share_dir;
}

1;

=back

=head1 AUTHOR

Lawrence Murray <lawrence.murray@csiro.au>

=head1 VERSION

$Rev$ $Date$
