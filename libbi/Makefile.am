AUTOMAKE_OPTIONS = subdir-objects

# compile and link flags
AM_CPPFLAGS = $(OPENMP_CPPFLAGS)
AM_CXXFLAGS = $(OPENMP_CXXFLAGS)
AM_LDFLAGS = $(OPENMP_LDFLAGS)
ACLOCAL_AMFLAGS = -I m4

# CUDA files setup
CUDA = nvcc
CUDAFLAGS = -w -arch sm_13 -Xcompiler="$(OPENMP_CXXFLAGS)"
#  -Xptxas="-v"

# libraries
lib_LTLIBRARIES = libbi.la
libbi_la_SOURCES = \
  src/bi/bi.cpp \
  src/bi/model/Coord.cpp \
  src/bi/buffer/KalmanFilterNetCDFBuffer.cpp \
  src/bi/buffer/NetCDFBuffer.cpp \
  src/bi/buffer/OptimiserNetCDFBuffer.cpp \
  src/bi/buffer/ParticleFilterNetCDFBuffer.cpp \
  src/bi/buffer/ParticleMCMCNetCDFBuffer.cpp \
  src/bi/buffer/SimulatorNetCDFBuffer.cpp \
  src/bi/buffer/SparseInputBuffer.cpp \
  src/bi/buffer/SparseInputNetCDFBuffer.cpp \
  src/bi/host/shared_host.cpp \
  src/bi/host/math/cblas.cpp \
  src/bi/host/math/lapack.cpp \
  src/bi/host/math/qrupdate.cpp \
  src/bi/host/ode/IntegratorConstants.cpp \
  src/bi/method/MetropolisResampler.cpp \
  src/bi/method/MultinomialResampler.cpp \
  src/bi/method/StratifiedResampler.cpp \
  src/bi/misc/omp.cpp \
  src/bi/random/Random.cpp
  
if ENABLE_SSE
libbi_la_SOURCES += \
  src/bi/sse/sse_shared_host.cpp
endif
  
if ENABLE_GPU
libbi_la_SOURCES += \
  src/bi/cuda/device.cu \
  src/bi/cuda/math/magma.cu \
  src/bi/cuda/math/cublas.cu \
  src/bi/cuda/random/curand.cu
endif

include_HEADERS = \
  src/bi/bi.hpp
  
# other
dist_noinst_SCRIPTS = autogen.sh

# command line options
if ENABLE_DOUBLE
CUDAFLAGS += -DENABLE_DOUBLE
CXXFLAGS += -DENABLE_DOUBLE
endif

if ENABLE_GPU
CUDAFLAGS += -DENABLE_GPU
CXXFLAGS += -DENABLE_GPU -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
else
CUDAFLAGS += -Xcompiler -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
CXXFLAGS += -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
endif

if ENABLE_SSE
CXXFLAGS += -DENABLE_SSE -msse3
endif

# suffix rules for CUDA files
.cu.o:
	$(CUDA) -c $(CUDAFLAGS) $(CPPFLAGS) -o $@ $<

.cu.lo:
	$(LIBTOOL) --tag=CC --mode=compile ./nvcc_wrapper.pl $(CUDA) -c $(CUDAFLAGS) $(CPPFLAGS) -o $@ $<
