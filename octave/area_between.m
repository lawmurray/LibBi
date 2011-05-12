% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} area_between (x, y1, y2, c)
%
% Plot area between two curves.
%
% @itemize
% @bullet{ @var{x} X-coordinates.}
% @bullet{ @var{y1} Y-coordinates of first curve.}
% @bullet{ @var{y2} Y-coordinates of second curve.}
% @bullet{ @var{c} Color.}
% @end itemize
% @end deftypefn
%
function area_between (x, y1, y2, c)
    % check arguments
    if (nargin < 3 || nargin > 4)
        print_usage ();
    end
    if (!isvector(x))
        error ('x must be vector');
    end
    if (!isvector(y1))
        error ('y1 must be vector');
    end
    if (!isvector(y2))
        error ('y2 must be vector');
    end
    
    % plot
    a = [x x(end:-1:1)];
    b = [y1 y2(end:-1:1)];
        
    if (nargin == 4)
        patch(a, b, fade(c, 0.5), 'linewidth', 1, 'edgecolor', c);
    else
        patch(a, b, 'linewidth', 1, 'edgecolor', c);
    end
end
