% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1687 $
% $Date: 2011-06-28 11:46:45 +0800 (Tue, 28 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} {@var{n} = } num_dims (@var{nc}, @var{name})
%
% Number of dimensions for variable in NetCDF file.
%
% @itemize
% @bullet{ @var{nc} NetCDF file handle.}
%
% @bullet{ @var{name} Name of the variable.}
% @end itemize
% @end deftypefn
%
function n = ncnumdims (nc, name)
    n = length (ncdim (nc{name}));
end
