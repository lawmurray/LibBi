% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} {@var{c} =} watercolour (@var{i})
%
% Retrieve colour from watercolour palette.
%
% @itemize
% @bullet{ @var{i} Colour index.}
% @end itemize
%
% Returns colour at given index.
%
% @end deftypefn
%
function c = watercolour (i, fade)
    % check arguments
    if (nargin != 1)
        print_usage ();
    end
    if (!isscalar(i))
        error ('i must be scalar');
    end
    
    % full palette
    palette = [0.3373, 0.7059, 0.9137;
               0.9020, 0.6235, 0.0000;
               0.0000, 0.6196, 0.4510;
               0.9412, 0.8941, 0.2588;
               0.0000, 0.4471, 0.6980;
               0.8353, 0.3686, 0.0000;
               0.8000, 0.4745, 0.6549];
    
    % select from palette
    c = palette(mod(i - 1, rows(palette)) + 1, :);
end
