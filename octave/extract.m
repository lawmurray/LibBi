% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} extract (@var{in}, @var{out}, @var{vars})
%
% Extract variables from one NetCDF file into another.
%
% @itemize
% @bullet{ @var{in} Input file. Name of a NetCDF file output by simulate.}
%
% @bullet{ @var{out} Output file. Name of a NetCDF file to create.}
%
% @bullet{ @var{vars} Names of variables as cell array.}
% @end itemize
% @end deftypefn
%
function extract (in, out, vars)
    % check arguments
    if nargin < 3
        print_usage ();
    end

    % input file
    nci = netcdf(in, 'r');
    
    % output file
    nco = netcdf(out, 'c', '64bit-offset');

    % dimensions
    dims = ncdim (nci);
    for i = 1:length(dims)
        dim = ncname (dims{i});
        nco(dim) = length (nci(dim));
    end
 
    % variables
    for i = 1:length(vars)
        var = vars{i};
        if ncvarexists (nci, var)
            dims = ncdim (nci{var});
            numdims = length (dims);
            switch numdims
              case 1
                nco{var} = ncdouble (ncname (dims{1}));
                nco{var}(:) = nci{var}(:);
              case 2
                nco{var} = ncdouble (ncname (dims{1}), ncname (dims{2}));
                nco{var}(:) = nci{var}(:,:);
              case 3
                nco{var} = ncdouble (ncname (dims{1}), ncname (dims{2}), ...
                    ncname (dims{3}));
                nco{var}(:) = nci{var}(:,:,:);
              case 4
                nco{var} = ncdouble (ncname (dims{1}), ncname (dims{2}), ...
                    ncname (dims{3}), ncname (dims{4}));
                nco{var}(:) = nci{var}(:,:,:,:);
              case 5
                nco{var} = ncdouble (ncname (dims{1}), ncname (dims{2}), ...
                    ncname (dims{3}), ncname (dims{4}), ncname (dims{5}));
                nco{var}(:) = nci{var}(:,:,:,:,:);
            end
        end
    end
    
    ncclose(nci);
    ncclose(nco);
end
