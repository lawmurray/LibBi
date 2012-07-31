#!/usr/bin/env perl

use strict;
use warnings;

use File::Spec;
use File::Slurp;
use File::Temp qw(tempdir);

my $ACTION_DIR = File::Spec->catdir('lib', 'Bi', 'Action');
my $BLOCK_DIR = File::Spec->catdir('lib', 'Bi', 'Block');
my $CLIENT_DIR = File::Spec->catdir('lib', 'Bi', 'Client');

my $ACTION_TMPDIR = tempdir(CLEANUP => 1);
my $BLOCK_TMPDIR = tempdir(CLEANUP => 1);
my $CLIENT_TMPDIR = tempdir(CLEANUP => 1);

my $ACTION_OUT = File::Spec->catfile('docs', 'tex', 'actions.tex');
my $BLOCK_OUT = File::Spec->catfile('docs', 'tex', 'blocks.tex');
my $CLIENT_OUT = File::Spec->catfile('docs', 'tex', 'clients.tex');
my $TEX_DIR = File::Spec->catdir('docs', 'tex');
my $HTML_DIR = File::Spec->catdir('docs', 'html');

# copy pods to temporary directory
`cp $ACTION_DIR/*.pm $ACTION_TMPDIR`;
`cp $BLOCK_DIR/*.pm $BLOCK_TMPDIR`;
`cp $CLIENT_DIR/*.pm $CLIENT_TMPDIR`;

# preprocess pod
foreach my $dir ($ACTION_TMPDIR, $BLOCK_TMPDIR, $CLIENT_TMPDIR) {
    my @files = <$dir/*.pm>;	
    foreach my $file (@files) {
	my $contents = read_file($file);
	$contents = process_pod($contents);
	write_file($file, $contents);
    }
}

# convert pod to tex
system("pod2latex -h1level 2 -modify -sections '!(AUTHOR|VERSION|INHERITS|METHODS)' -out $ACTION_OUT `ls $ACTION_TMPDIR/*[A-Za-z0-9].pm | sort`");
system("pod2latex -h1level 2 -modify -sections '!(AUTHOR|VERSION|INHERITS|METHODS)' -out $BLOCK_OUT `ls $BLOCK_TMPDIR/*[A-Za-z0-9].pm | sort`");
system("pod2latex -h1level 2 -modify -sections '!(AUTHOR|VERSION|INHERITS|METHODS)' -out $CLIENT_OUT `ls $CLIENT_TMPDIR/*[A-Za-z0-9].pm | sort`");

# postprocess tex
foreach my $file ($ACTION_OUT, $BLOCK_OUT, $CLIENT_OUT) {
    my $contents = read_file($file);
    $contents = &process_tex($contents);
    write_file($file, $contents);
}

# convert tex to html
system("mkdir -p $HTML_DIR");
system("cd $TEX_DIR; latex2html -antialias -antialias_text -show_section_numbers -local_icons -toc_depth 4 -split 5 -dir ../html index.tex");

# compile tex
system("cd $TEX_DIR; pdflatex index.tex; makeindex index; pdflatex index.tex; pdflatex index.tex");

sub process_tex {
    my $contents = shift;

    $contents =~ s/\\texttt\{/\{\\sf /g;
    $contents =~ s/\\begin\{verbatim\}((?:.|\n)*?)\\end\{verbatim\}/&process_tt($1)/gem;
    $contents =~ s/I<(.*?)>/\\textsl\{$1\}/g;
    $contents =~ s/(SYNOPSIS|DESCRIPTION|ARGUMENTS|PARAMETERS|OPTIONS|SEE ALSO)/ucfirst(lc($1))/ge;
    $contents =~ s/\$\\backslash\$hyperref\[hyper\]\[([A-Za-z0-9\-\_\\]*?)\]\\\{([A-Za-z0-9\-\_\\]*?)\\\}/"\\hyperref\[hyper\]\[" . &process_label($1) . "\]\{$2\}\\index\{" . &process_index($2) . "\}"/ge;
    $contents =~ s/\\label\{([A-Za-z0-9\-\_\\]*?)\}/"\\label\{" . &process_label($1) . "\}"/ge;
    $contents =~ s/\\index\{([A-Za-z0-9\-\_\\]+?)\}/"\\index\{" . &process_index($1) . "\}"/ge;

    # remove index entries for subsections and paragraphs
    $contents =~ s/\\index\{[A-Za-z0-9\-\_\\ ]+?\!(?:SYNOPSIS|DESCRIPTION|ARGUMENTS|PARAMETERS|OPTIONS|SEE ALSO)\}//ig;
    $contents =~ s/(\\paragraph\*?\{.*?)\\index\{.*?\}/$1/g;

    $contents =~ s/\\subsection{((?:\w|\\_)+)/\\subsection{{\\sf $1}/g;
    
    return $contents;
}

sub process_tt {
    my $contents = shift;
	
    $contents =~ s/([{}])/\\$1/g;
    $contents =~ s/\.\.\./\\(\\ldots\\)/g;
    
    return "\\begin\{alltt\}\{\\sf$contents\}\\end\{alltt\}";
}

sub process_label {
    my $contents = shift;
	
    $contents =~ s/\\_/_/g;
    $contents =~ s/-/_/g;
	
    return $contents;
}

sub process_index {
    my $contents = shift;
	
    $contents =~ s/(?:<\\)_/\\_/g;
    $contents =~ s/-/_/g;
	
    return $contents;
}

sub process_pod {
    my $contents = shift;
	
    $contents =~ s/L\<(\w+)\>/C\<\\hyperref\[hyper\]\[$1\]\{$1\}\>/g;
    $contents =~ s/L\<(\w+)\|(\w+)\>/C\<\\hyperref\[hyper\]\[$2\]\{$1\}\>/g;
	
    return $contents;
}
