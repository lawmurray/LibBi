% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} area_between (x, y1, y2, c, fd, alpha)
%
% Plot area between two curves.
%
% @itemize
% @bullet{ @var{x} X-coordinates.}
% @bullet{ @var{y1} Y-coordinates of first curve.}
% @bullet{ @var{y2} Y-coordinates of second curve.}
% @bullet{ @var{c} Color.}
% @bullet{ @var{fd} Opaque fade.}
% @bullet{ @var{alpha} Alpha.}
% @end itemize
% @end deftypefn
%
function area_between (x, y1, y2, c, fd, alpha)
    % check arguments
    if (nargin < 4 || nargin > 6)
        print_usage ();
    end
    if nargin < 5
        fd = 0.5;
    end
    if nargin < 6
        alpha = 1.0;
    end
    if !isvector(x)
        error ('x must be vector');
    end
    if !isvector(y1)
        error ('y1 must be vector');
    end
    if !isvector(y2)
        error ('y2 must be vector');
    end
    
    % plot
    a = [x x(end:-1:1)];
    b = [y1 y2(end:-1:1)];
        
    bg = fade(c, fd);
    patch(a, b, bg, 'linewidth', 1, 'edgecolor', c, 'facealpha', alpha);
end
