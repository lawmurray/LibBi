% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1367 $
% $Date: 2011-04-01 13:10:49 +0800 (Fri, 01 Apr 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} {@var{c} =} fade (@var{rgb}, @var{amount})
%
% Fade colour.
%
% @itemize
% @bullet{ @var{rgb} RGB triplet.}
%
% @bullet{ @var{amount} Amount to fade.}
% @end itemize
%
% Returns colour faded by the given amount.
%
% @end deftypefn
%
function c = fade (rgb, amount)
    % check arguments
    if (nargin < 1 || nargin > 2)
        print_usage ();
    end
    if (nargin < 2)
        amount = 1.0;
    end
    if length(rgb) != 3
        error ('rgb must be three component RGB triplet');
    end
    
    c = amount*rgb + (1.0 - amount);
end
