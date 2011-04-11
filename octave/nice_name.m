% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1371 $
% $Date: 2011-04-04 13:51:32 +0800 (Mon, 04 Apr 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} {@var{nice} = } nice_name (@var{name}, @var{offsets})
%
% Construct human-readable name for variable.
%
% @itemize
% @bullet{ @var{name} Name of the variable.}
%
% @bullet{ @var{offsets} zero to three element vector containing dimension
% indices for the desired component of this variable.
% @end itemize
% @end deftypefn
%
function nice = nice_name(name, offsets)
    % check arguments
    if nargin < 1 || nargin > 2
        print_usage ();
    elseif nargin == 1
        offsets = [];
    elseif nargin == 2 && length(offsets) > 3
        error ('offsets must be vector of length zero to three');
    end
    
    nice = name;
    if length(offsets) > 0
        nice = strcat(nice, '[');
    end
    for i = 1:length(offsets)
        nice = strcat(nice, num2str(offsets(i)));
        if i != length(offsets)
            nice = strcat(nice, ',');
        end
    end
    if length(offsets) > 0
        nice = strcat(nice, ']');
    end
end

