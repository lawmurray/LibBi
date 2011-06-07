% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1410 $
% $Date: 2011-04-15 12:35:42 +0800 (Fri, 15 Apr 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} @var{valid} = } check_coord (@var{coord})
%
% Check coord argument given to function.
%
% @itemize
% @bullet{ @var{coord} Vector of spatial coordinates of zero
% to three elements, giving the x, y and z coordinates of a
% variable.}
% @end itemize
% @end deftypefn
%
function valid = check_coord (coord)
    valid = isempty (coord) || (isvector (coord) && length (coord) <= 3);
end
