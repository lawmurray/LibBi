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
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% simulate.}
%
% @bullet{ @var{invars} Cell array of strings naming the variables
% of this file to disturb.}
%
% @bullet{ @var{out} Output file. Name of a NetCDF file to create.}
%
% @bullet{ @var{outvars} Cell array of strings naming the variables
% to create in the output file. Its length must match that of
% @var{invars}.}
%
% @bullet{ @var{p} Index along the @t{nr} dimension of the
% input file, indicating the trajectory to disturb to create
% observations.}
%
% @bullet{ @var{S} List of standard deviations of disturbance
% noise. Each value produces a corresponding record along the @t{ns}
% dimension of the output file.}
%
% @bullet{ @var{logn} True for log-normal noise, false for normal noise.}
% @end itemize
% @end deftypefn
%
function twindata (in, invars, out, outvars, p, S, logn)
    % check arguments
    if (nargin < 6)
        print_usage ();
    end
    if (length (invars) != length (outvars))
        error ('Lengths of invars and outvars must match');
    end
    if (!isscalar (p))
        error ('p must be scalar');
    end
    if (!columns (S) == 1)
        error ('S must be scalar or column vector');
    end
    
    % input file
    nci = netcdf(in, 'r');
    
    % output file
    nco = netcdf(out, 'c');
    nco('ns') = length(S);
    nco('nr') = nci('nr')(:);
    nco('np') = 1;
    nco{'time'} = ncdouble('nr');
    %nco{'time'}(:) = nci{'time'}(:);

    % construct data
    for i = 1:length(invars)
        x = nci{invars{i}}(:,p)';
        U = normrnd(0.0, 1.0, length(S), length(x));
        if logn
            Y = exp(repmat(log(x), length(S), 1) + repmat(S, 1, length(x)).*U);
        else
            Y = repmat(x, length(S), 1) + repmat(S, 1, length(x)).*U;
        end
        
        nco{outvars{i}} = ncdouble('ns', 'nr');
        nco{outvars{i}}(:,:) = Y;
    end
    
    ncclose(nci);
    ncclose(nco);
end
