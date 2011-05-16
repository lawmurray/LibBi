% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1500 $
% $Date: 2011-05-16 11:49:58 +0800 (Mon, 16 May 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} {@var{exists} = } ncvarexists (@var{nc}, @var{var})
%
% Check if a dimension of the given name exists in the given NetCDF file.
%
% @itemize
% @bullet{ @var{nc} NetCDF file hande.}
%
% @bullet( @var{var} Variable name.}
% @end itemize
% @end deftypefn
%
function exists = ncvarexists (nc, var)
    % check arguments
    if (nargin != 2)
        print_usage ();
    end

    exists = 0;
    for i = 1:length(ncvar(nc))
        if strcmp(var, ncname(ncvar(nc){i}))
            exists = 1;
        end
    end
end
