##
## Generate a Makefile for compilation.
##
## @author Lawrence Murray <lawrence.murray@csiro.au>
## $Rev: 1309 $
## $Date: 2011-02-25 16:22:39 +0800 (Fri, 25 Feb 2011) $
##

use FindBin qw($Bin);

# Settings
$NAME = $ARGV[0];
$SPEC = $NAME;
$SRCDIR = 'src';
$BUILDDIR = 'build';
$CPPDIR = "$SRCDIR/model";
$OLDDIR = 'old';

# Compilers
$GCC = 'g++';
$ICC = 'icpc';
$CUDACC = 'nvcc';

# Common compile flags
$CPPINCLUDES = "-I$Bin/../../lib/src -I/tools/cuda/3.1/cuda/include/ -I/usr/local/cuda/include -I/usr/local/include/thrust -I/tools/magma/0.2/include -I/usr/local/atlas/include";
$CXXFLAGS = "-Wall `nc-config --cflags` $CPPINCLUDES";
$CUDACCFLAGS = "-arch sm_13 -Xptxas=\"-v\" -Xcompiler=\"-Wall -fopenmp\" `nc-config --cflags` $CPPINCLUDES";
$LINKFLAGS = "-L\"$Bin/../../lib/build\" -L\"/tools/magma/0.2/lib\" -L\"/tools/boost/1.43.0/lib\" -lbi -lmagma -lmagmablas -lgfortran -lnetcdf_c++ `nc-config --libs` -lpthread";
# ^ may need f2c, g2c or nothing in place of gfortran
$DEPFLAGS = '-I"../bi/src"'; # flags for dependencies check

# GCC options
$GCC_CXXFLAGS = '-fopenmp -Wno-parentheses';
$GCC_LINKFLAGS = '-lgomp';

# Intel C++ compiler options
$ICC_CXXFLAGS = '-openmp -malign-double -wd424 -wd981 -wd383 -wd1572 -wd869 -wd304 -wd444 -wd1418 -wd1782';
$ICC_LINKFLAGS = '-openmp';

# Math library option flags
$ATLAS_LINKFLAGS = '-L/usr/local/atlas/lib -llapack -lptf77blas -lptcblas -latlas -lm'; # "pt" prefix for multithreaded versions
$MKL_LINKFLAGS = '-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core';
$MATH_LINKFLAGS = '-lblas -lcblas -llapack -lm';

# Release flags
$RELEASE_CXXFLAGS = ' -O3 -funroll-loops -fomit-frame-pointer -g';
$RELEASE_CUDACCFLAGS = ' -O3 -Xcompiler="-O3 -funroll-loops -fomit-frame-pointer -g"';
$RELEASE_LINKFLAGS = ' -lcublas -lcurand -lcudart';

# Debugging flags
$DEBUG_CXXFLAGS = ' -g';
$DEBUG_CUDACCFLAGS = ' -g';
$DEBUG_LINKFLAGS = ' -lcublas -lcurand -lcudart';

# Profiling flags
$PROFILE_CXXFLAGS = ' -O1 -pg -g3';
$PROFILE_CUDACCFLAGS = ' -O1 --compiler-options="-O1 -pg -g"';
$PROFILE_LINKFLAGS = ' -pg -g3 -lcublas -lcurand -lcudart';

# Disassembly flags
$DISASSEMBLE_CUDACCFLAGS = ' -keep';
$DISASSEMBLE_LINKFLAGS = ' -lcublas -lcurand -lcudart';

# Ocelot flags
$OCELOT_CXXFLAGS = ' -g -O3 -funroll-loops';
$OCELOT_CUDACCFLAGS = ' -g -O3 -Xcompiler="-O3 -funroll-loops"';
$OCELOT_LINKFLAGS = ' -locelot -lhydralize -lcublas -lcurand';

# Walk through source
@files = ($SRCDIR);
while (@files) {
  $file = shift @files;
  if (-d $file) {
    # recurse into directory
    opendir(DIR, $file);
    push(@files, map { "$file/$_" } grep { !/^\./ } readdir(DIR));
    closedir(DIR);
  } elsif (-f $file && $file =~ /\.(cu|c|cpp)$/) {
    $ext = $1;

    # target name
    $target = $file;
    $target =~ s/^$SRCDIR/$BUILDDIR/;
    $target =~ s/\.\w+$/.$ext.o/;

    # determine compiler and appropriate flags
    if ($file =~ /\.cu$/) {
      $cc = $CUDACC;
      $ccstr = "\$(CUDACC)";
      $flags = $CUDACCFLAGS;
      $flagstr = "\$(CUDACCFLAGS)";
    } else {
      $cc = $GCC;
      $ccstr = "\$(CXX)";
      $flags = $CXXFLAGS;
      $flagstr = "\$(CXXFLAGS)";
    }
    
    # determine dependencies of this source and construct Makefile target
    $target =~ /(.*)\//;
    $dir = $1;
    $dirs{$dir} = 1;

    $target =~ s/$BUILDDIR/\$\(BUILDDIR\)/;
    $command = `$cc $flags -M $file`;
    $command =~ s/.*?\:\w*//;
    $command = "$target: " . $command;
    $command .= "\tmkdir -p $dir\n";
    $command .= "\t$ccstr -o \$\@ $flagstr -c \$<\n";
    #$command .= "\trm -f *.linkinfo\n";
    push(@targets, $target);
    push(@commands, $command);
    if ($dir eq "$BUILDDIR/model") {
      push(@models, $target);
    }
  }
}

# Write Makefile
print <<End;
ifdef USE_CONFIG
include config.mk
endif

NAME=$NAME
SPEC=$SPEC

BUILDDIR=$BUILDDIR
SRCDIR=$SRCDIR
CPPDIR=$CPPDIR
OLDDIR=$OLDDIR

CXXFLAGS=$CXXFLAGS
LINKFLAGS=$LINKFLAGS
CUDACC=$CUDACC
CUDACCFLAGS=$CUDACCFLAGS

GCC_CXXFLAGS=$GCC_CXXFLAGS
GCC_LINKFLAGS=$GCC_LINKFLAGS
ICC_CXXFLAGS=$ICC_CXXFLAGS
ICC_LINKFLAGS=$ICC_LINKFLAGS

ATLAS_LINKFLAGS=$ATLAS_LINKFLAGS
MKL_LINKFLAGS=$MKL_LINKFLAGS
MATH_LINKFLAGS=$MATH_LINKFLAGS

DEBUG_CXXFLAGS=$DEBUG_CXXFLAGS
DEBUG_CUDACCFLAGS=$DEBUG_CUDACCFLAGS
DEBUG_LINKFLAGS=$DEBUG_LINKFLAGS

RELEASE_CXXFLAGS=$RELEASE_CXXFLAGS
RELEASE_CUDACCFLAGS=$RELEASE_CUDACCFLAGS
RELEASE_LINKFLAGS=$RELEASE_LINKFLAGS

PROFILE_CXXFLAGS=$PROFILE_CXXFLAGS
PROFILE_CUDACCFLAGS=$PROFILE_CUDACCFLAGS
PROFILE_LINKFLAGS=$PROFILE_LINKFLAGS

DISASSEMBLE_CXXFLAGS=$DISASSEMBLE_CXXFLAGS
DISASSEMBLE_CUDACCFLAGS=$DISASSEMBLE_CUDACCFLAGS
DISASSEMBLE_LINKFLAGS=$DISASSEMBLE_LINKFLAGS

OCELOT_CXXFLAGS=$OCELOT_CXXFLAGS
OCELOT_CUDACCFLAGS=$OCELOT_CUDACCFLAGS
OCELOT_LINKFLAGS=$OCELOT_LINKFLAGS

ifdef USE_INTEL
CXX=$ICC
LINKER=$ICC
CXXFLAGS += \$(ICC_CXXFLAGS)
LINKFLAGS += \$(ICC_LINKFLAGS)
else
CXX=$GCC
LINKER=$GCC
CXXFLAGS += \$(GCC_CXXFLAGS)
LINKFLAGS += \$(GCC_LINKFLAGS)
endif

ifdef USE_DOUBLE
CUDACCFLAGS += -DUSE_DOUBLE
CXXFLAGS += -DUSE_DOUBLE
else
ifdef USE_FAST_MATH
CUDACCFLAGS += -use_fast_math
endif
endif

ifdef USE_CPU
CUDACCFLAGS += -DUSE_CPU -Xcompiler -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP
CXXFLAGS += -DUSE_CPU -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP
EXT=cpp
else
EXT=cu
endif

ifdef USE_SSE
CXXFLAGS += -DUSE_SSE -msse3
CUDACCFLAGS += -DUSE_SSE
endif

ifdef USE_LOCAL
CUDACCFLAGS += -DUSE_LOCAL
CXXFLAGS += -DUSE_LOCAL
endif

ifdef USE_MKL
CXXFLAGS += -DUSE_MKL
CUDACCFLAGS += -DUSE_MKL
endif

ifdef USE_ATLAS
LINKFLAGS += \$(ATLAS_LINKFLAGS)
else
ifdef USE_MKL
LINKFLAGS += \$(MKL_LINKFLAGS)
else
LINKFLAGS += \$(MATH_LINKFLAGS)
endif
endif

ifdef USE_DOPRI5
CUDACCFLAGS += -DUSE_DOPRI5
CXXFLAGS += -DUSE_DOPRI5
endif

ifdef USE_TEXTURE
CUDACCFLAGS += -DUSE_TEXTURE
CXXFLAGS += -DUSE_TEXTURE
endif

ifdef USE_RIPEN
CUDACCFLAGS += -DUSE_RIPEN
CXXFLAGS += -DUSE_RIPEN
endif

ifdef NDEBUG
CUDACCFLAGS += -DNDEBUG
CXXFLAGS += -DNDEBUG
endif

ifdef DEBUG
CUDACCFLAGS += \$(DEBUG_CUDACCFLAGS)
CXXFLAGS += \$(DEBUG_CXXFLAGS)
LINKFLAGS += \$(DEBUG_LINKFLAGS)
endif

ifdef RELEASE
CUDACCFLAGS += \$(RELEASE_CUDACCFLAGS)
CXXFLAGS += \$(RELEASE_CXXFLAGS)
LINKFLAGS += \$(RELEASE_LINKFLAGS)
endif

ifdef PROFILE
CUDACCFLAGS += \$(PROFILE_CUDACCFLAGS)
CXXFLAGS += \$(PROFILE_CXXFLAGS)
LINKFLAGS += \$(PROFILE_LINKFLAGS)
endif

ifdef DISASSEMBLE
CUDACCFLAGS += \$(DISASSEMBLE_CUDACCFLAGS)
CXXFLAGS += \$(DISASSEMBLE_CXXFLAGS)
LINKFLAGS += \$(DISASSEMBLE_LINKFLAGS)
endif

ifdef OCELOT
CUDACCFLAGS += \$(OCELOT_CUDACCFLAGS)
CXXFLAGS += \$(OCELOT_CXXFLAGS)
LINKFLAGS += \$(OCELOT_LINKFLAGS)
endif

ifdef USE_CPU_ODE
CUDACCFLAGS += -DUSE_CPU_ODE
CXXFLAGS += -DUSE_CPU_ODE
endif

End

# Default targets
print <<End;
default: \$(BUILDDIR)/simulate \$(BUILDDIR)/pf \$(BUILDDIR)/ukf \$(BUILDDIR)/urts \$(BUILDDIR)/mcmc \$(BUILDDIR)/likelihood

End

# Artifacts
my $models = join(' ', @models);

print "\$(BUILDDIR)/simulate: \$(BUILDDIR)/simulate.\$(EXT).o $models\n";
print "\t\$(LINKER) -o \$\@ \$^ \$(LINKFLAGS)\n\n";

print "\$(BUILDDIR)/pf: \$(BUILDDIR)/pf.\$(EXT).o $models\n";
print "\t\$(LINKER) -o \$\@ \$^ \$(LINKFLAGS)\n\n";

print "\$(BUILDDIR)/ukf: \$(BUILDDIR)/ukf.\$(EXT).o $models\n";
print "\t\$(LINKER) -o \$\@ \$^ \$(LINKFLAGS)\n\n";

print "\$(BUILDDIR)/urts: \$(BUILDDIR)/urts.\$(EXT).o $models\n";
print "\t\$(LINKER) -o \$\@ \$^ \$(LINKFLAGS)\n\n";

print "\$(BUILDDIR)/mcmc: \$(BUILDDIR)/mcmc.\$(EXT).o \$(BUILDDIR)/device.cu.o $models\n";
print "\t\$(LINKER) -o \$\@ \$^ \$(LINKFLAGS)\n\n";

print "\$(BUILDDIR)/likelihood: \$(BUILDDIR)/likelihood.\$(EXT).o \$(BUILDDIR)/device.cu.o $models\n";
print "\t\$(LINKER) -o \$\@ \$^ \$(LINKFLAGS)\n\n";

# Targets
print join("\n", @commands);
print "\n";

# Build directory tree targets
foreach $dir (sort keys %dirs) {
  print "$dir:\n\tmkdir -p $dir\n\n";
}

# Clean target
print <<End;
clean:
\trm -rf \$(BUILDDIR)
End

# Clobber target
print <<End;
clobber: clean
\trm -rf Makefile config.mk *.sh \$(NAME).pdf \$(BUILDDIR) \$(SRCDIR) \$(OLDDIR)
End

# Save config target
print <<End;
save:
\tmkdir -p \$(OLDDIR)
\tcp -f config.mk config.sh \$(OLDDIR)
End

# Restore config target
print <<End;
restore:
\tcp -f \$(OLDDIR)/config.mk \$(OLDDIR)/config.sh .
End
