% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1371 $
% $Date: 2011-04-04 13:51:32 +0800 (Mon, 04 Apr 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} {@var{X} = } read_var (@var{nc}, @var{name}, @var{coord})
%
% Read variable from NetCDF file.
%
% @itemize
% @bullet{ @var{nc} NetCDF file handle.}
%
% @bullet{ @var{name} Name of the variable.}
%
% @bullet{ @var{ps} Trajectory indices.
%
% @bullet{ @var{coord} Spartial coordinates. Zero to three element vector
% containing spatial coordinates for the desired component of this variable.
% @end itemize
% @end deftypefn
%
function X = read_var (nc, name, ps, coord)
    % check arguments
    if nargin < 2 || nargin > 4
        print_usage ();
    elseif nargin == 2
        ps = [1];
        coord = [];
    elseif nargin == 3
        coord = [];
    elseif length (coord) > 3
        error ('coord must be vector of length zero to three');
    end
    
    % check number of dimensions
    numdims = ndims (nc{name});
        
    % read
    if numdims == 2 && length(coord) == 0
        X = nc{name}(ps);
    else
        if length(coord) + 2 != length(ncdim(nc{name}))
            error (sprintf('wrong number of dimensions given for %s', name));
        end
        if length(coord) == 0
            X = nc{name}(:,ps);
        elseif length(coord) == 1
            X = nc{name}(:,coord(1),ps);
        elseif length(coord) == 2
            X = nc{name}(:,coord(1),coord(2),ps);
        elseif length(coord) == 3
            X = nc{name}(:,coord(1),coord(2),coord(3),ps);
        end
    end
    X = squeeze(X);
end
