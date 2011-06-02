% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} gen_obs (@var{in}, @var{invar}, @var{out}, @var{outvar}, @var{p}, @var{S}, @var{logn}, @var{coords})
%
% Generate a data set for twin experiments from the output of the simulate
% program.
%
% @itemize
% @bullet{ @var{in} Input file. Name of a NetCDF file output by simulate.}
%
% @bullet{ @var{invar} Name of variable from input file to disturb.}
%
% @bullet{ @var{out} Output file. Name of a NetCDF file to create.}
%
% @bullet{ @var{outvar} Name of variable in output file to create.}
%
% @bullet{ @var{p} Trajectory index, indicating that to perturb to create
% observations.}
%
% @bullet{ @var{ts} Time indices.}
%
% @bullet{ @var{S} List of standard deviations of disturbance
% noise. Each value produces a corresponding record along the @t{ns}
% dimension of the output file.}
%
% @bullet{ @var{logn} (optional) True for log-normal noise, false for normal
% noise.}
%
% @bullet{ @var{coords} (optional) Matrix of spatial coordinates of zero
% to three columns. Each row gives the x, y and z coordinates of a
% component of @var{invar} to disturb.}
% @end itemize
% @end deftypefn
%
function gen_obs (in, invar, out, outvar, p, ts, S, logn, coords)
    % check arguments
    if (nargin < 7 || nargin > 9)
        print_usage ();
    end
    if (!ischar (invar))
        error ('invar must be a string');
    end
    if (!ischar (outvar))
        error ('outvar must be a string');
    end
    if (!isscalar (p))
        error ('p must be scalar');
    end
    if (!columns (S) == 1)
        error ('S must be scalar or column vector');
    end
    if nargin < 8
        logn = 0;
    end
    if nargin < 9
        coords = [];
        M = 1;
    elseif !ismatrix (coords) || columns (coords) > 3
        error ('coords should be a matrix with at most three columns');
    else
        M = rows (coords);
    end
        
    % input file
    nci = netcdf(in, 'r');
    T = nci('nr')(:);
    if length(ts) == 0
        ts = [1:T];
    end
    
    % output file
    if exist (out, "file")
        nco = netcdf(out, 'w');
    else
        nco = netcdf(out, 'c');
    end

    % dimensions
    rdim = sprintf('nr_%s', outvar);
    cdim = sprintf('nc_%s', outvar);
    
    nco('ns') = length (S);
    nco(rdim) = M*length(ts);
    if columns (coords) > 1
        nco(cdim) = columns (coords);
    end
    
    % time variable
    tvar = sprintf('time_%s', outvar);
    nco{tvar} = ncdouble(rdim);
    nco{tvar}(:) = repmat(nci{'time'}(ts)', M, 1)(:);
    
    % coordinate variable
    cvar = sprintf('coord_%s', outvar);
    if columns (coords) > 1
        nco{cvar} = ncdouble(rdim, cdim);
        nco{cvar}(:,:) = repmat(coords - 1, length(ts), 1);
    elseif columns (coords) > 0
        nco{cvar} = ncdouble(rdim);
        nco{cvar}(:) = repmat(coords(:) - 1, length(ts), 1);
    end

    % construct data
    nco{outvar} = ncdouble('ns', rdim);
    for j = 1:M
        coord = coords(j,:);
        x = read_var(nci, invar, coord, p, ts)';
        u = normrnd(0.0, 1.0, 1, length(x));
        U = repmat(u, length(S), 1);
        X = repmat(x, length(S), 1);
        if logn
            Y = exp(log(X) + repmat(S(:), 1, length(x)).*U);
        else
            Y = X + repmat(S(:), 1, length(x)).*U;
        end
        nco{outvar}(:,j:M:end) = Y;
    end    
    ncclose(nci);
    ncclose(nco);
end
