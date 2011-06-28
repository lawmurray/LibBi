% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} ncdimexists (@var{nc}, @var{dim})
%
% Check if a dimension of the given name exists in the given NetCDF file.
%
% @itemize
% @bullet{ @var{nc} NetCDF file hande.}
%
% @bullet( @var{dim} Dimension name.}
% @end itemize
% @end deftypefn
%
function exists = ncdimexists (nc, dim)
    % check arguments
    if (nargin != 2)
        print_usage ();
    end

    exists = 0;
    for i = 1:length(ncdim(nc))
        if strcmp(dim, ncname(ncdim(nc){i}))
            exists = 1;
        end
    end
end
