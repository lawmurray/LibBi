% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} extract (@var{in}, @var{out}, @var{vars}, @var{rang})
%
% Extract variables from one NetCDF file into another.
%
% @itemize
% @bullet{ @var{in} Input file. Name of a NetCDF file.}
%
% @bullet{ @var{out} Output file. Name of a NetCDF file to create.}
%
% @bullet{ @var{vars} Names of variables as cell array.}
%
% @bullet{ @var{rang} Range along time dimension to extract.
% @end itemize
% @end deftypefn
%
function extract (in, out, vars, rang)
    % check arguments
    if nargin < 3
        print_usage ();
    end
    if nargin < 4
        rang = [];
    end

    % input file
    nci = netcdf(in, 'r');
    
    % output file
    nco = netcdf(out, 'c', '64bit-offset');

    % dimensions
    dims = ncdim (nci);
    for i = 1:length(dims)
        dim = ncname (dims{i});
        if strcmp(dim, 'nr')
            T = length (nci(dim));
            if isempty (rang)
                rang = [1:T];
            end
            nco(dim) = length (rang);
        else
            nco(dim) = length (nci(dim));
        end
    end
 
    % variables
    for i = 1:length(vars)
        var = vars{i};
        if ncvarexists (nci, var)
            dims = ncdim (nci{var});
            numdims = length (dims);
            switch numdims
              case 0
                nco{var} = ncdouble();
                nco{var}(:) = nci{var}(:);
              case 1
                nco{var} = ncdouble (ncname (dims{1}));
                nco{var}(:) = nci{var}(:);
              case 2
                nco{var} = ncdouble (ncname (dims{1}), ncname (dims{2}));
                nco{var}(:,:) = nci{var}(rang,:);
              case 3
                nco{var} = ncdouble (ncname (dims{1}), ncname (dims{2}), ...
                    ncname (dims{3}));
                nco{var}(:,:,:) = nci{var}(rang,:,:);
              case 4
                nco{var} = ncdouble (ncname (dims{1}), ncname (dims{2}), ...
                    ncname (dims{3}), ncname (dims{4}));
                nco{var}(:,:,:,:) = nci{var}(rang,:,:,:);
              case 5
                nco{var} = ncdouble (ncname (dims{1}), ncname (dims{2}), ...
                    ncname (dims{3}), ncname (dims{4}), ncname (dims{5}));
                nco{var}(:,:,:,:,:) = nci{var}(rang,:,:,:,:);
              otherwise
                error (sprintf('Variable %s has greater than 5 dimensions', ...
                    var));
            end
        end
    end
    
    ncclose(nci);
    ncclose(nco);
end
