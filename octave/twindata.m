% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} twindata (@var{in}, @var{invars}, @var{out}, @var{outvars}, @var{p}, @var{S}, @var{logs})
%
% Generate a data set for twin experiments from the output of the simulate
% program.
%
% @itemize
% @bullet @var{in} Input file. Gives the name of a NetCDF file output by
% simulate.
%
% @bullet @var{invars} Cell array of strings naming the variables
% of this file to disturb.
%
% @bullet @var{out} Output file. Name of a NetCDF file to create.
%
% @bullet @var{outvars} Cell array of strings naming the variables
% to create in the output file. Its length must match that of
% @var{invars}.
%
% @bullet @var{p} Index along the @t{nr} dimension of the
% input file, indicating the trajectory to disturb to create
% observations.
%
% @bullet @var{S} List of standard deviations of disturbance
% noise. Each value produces a corresponding record along the @t{ns}
% dimension of the output file.
%
% @bullet @var{log} True for log-normal noise, false for normal noise.
% @end itemize
%
% @end deftypefn

function twindata (in, invars, out, outvars, p, S, log)
    % check arguments
    if (nargin < 6)
        print_usage ();
    end
    if (length (invars) != length (outvars))
        error ("Lengths of @var{invars} and @var{outvars} must ...
               match");
    end
    if (!isscalar (p))
        error ("@var{p} must be scalar");
    end
    
    % open files
    nci = netcdf(in, 'r');
    nco = netcdf(out, 'c');
    
    for i = 1:length(invars)
        x = nci{invars{i}}(:,p);
    end
end
