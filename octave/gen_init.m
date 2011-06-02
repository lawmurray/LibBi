% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1500 $
% $Date: 2011-05-16 11:49:58 +0800 (Mon, 16 May 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} gen_init (@var{in}, @var{invar}, @var{out}, @var{outvar}, @var{p})
%
% Generate a initialisation file to match initial value of given trajectory
% in output of simulate program.
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
% @bullet{ @var{p} Index along the @t{np} dimension of the
% input file, indicating the trajectory to use in initialisation.}
% @end itemize
% @end deftypefn
%
function gen_init (in, invar, out, outvar, p)
    % check arguments
    if (nargin != 5)
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
        
    % input file
    nci = netcdf(in, 'r');
    
    % output file
    nco = netcdf(out, 'nc');
    
    % construct dimensions if necessary
    ndims = length(ncdim(nci{invar})) - 2;
    if ndims >= 1 && !ncdimexists(nco, 'nx')
        nco('nx') = length(nci('nx'));
    end
    if ndims >= 2 && !ncdimexists(nco, 'ny')
        nco('ny') = length(nci('ny'));
    end
    if ndims == 3 && !ncdimexists(nco, 'nz')
        nco('nz') = length(nci('nz'));
    end
    
    % construct variable
    if ndims == 0
        if !ncvarexists(nco, outvar)
            nco{outvar} = ncdouble();
        end
        nco{outvar}(:) = nci{invar}(1,p);
    elseif ndims == 1
        if !ncvarexists(nco, outvar)
            nco{outvar} = ncdouble('nx');
        end
        nco{outvar}(:) = nci{invar}(1,:,p);
    elseif ndims == 2
        if !ncvarexists(nco, outvar)
            nco{outvar} = ncdouble('ny', 'nx');
        end
        nco{outvar}(:,:) = nci{invar}(1,:,:,p);
    elseif ndims == 3
        if !ncvarexists(nco, outvar)
            nco{outvar} = ncdouble('nz', 'ny', 'nx');
        end
        nco{outvar}(:,:,:) = nci{invar}(1,:,:,:,p);
    end

    ncclose(nci);
    ncclose(nco);
end
