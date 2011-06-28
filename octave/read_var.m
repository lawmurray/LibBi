% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} {@var{X} = } read_var (@var{nc}, @var{name}, @var{coord}, @var{ps}, @var{ts})
%
% Read variable from NetCDF file.
%
% @itemize
% @bullet{ @var{nc} NetCDF file handle.}
%
% @bullet{ @var{name} Name of the variable.}
%
% @bullet{ @var{coord} (optional) Spatial coordinates. Zero to three
% element vector containing spatial coordinates for the desired component
% of this variable.}
%
% @bullet{ @var{ps} (optional) Trajectory indices.}
%
% @bullet{ @var{ts} (optional) Time indices.}
% @end itemize
% @end deftypefn
%
function X = read_var (nc, name, coord, ps, ts)
    % check arguments
    if nargin < 2 || nargin > 5
        print_usage ();
    elseif nargin == 2
        coord = [];
        ps = [];
        ts = [];
    elseif nargin == 3
        ps = [];
        ts = [];
    elseif nargin == 4
        ts = [];
    end
    
    if length (coord) > 3
        error ('coord must be vector of length zero to three');
    end
    
    % check number of dimensions
    numdims = length (ncdim (nc{name}));
    if ncdimexists (nc, 'nr')
        T = length (nc('nr'));    
    else
        T = 1;
    end
    if ncdimexists (nc, 'np')
        P = length (nc('np'));
    else
        error ('read_var only for files with np dimension');
    end
    
    if length (ps) == 0
        ps = [1:P];
    end
    if length (ts) == 0
        ts = [1:T];
    end
    
    % read
    if numdims == 1
        X = nc{name}(ps);
    elseif numdims == 2
        X = nc{name}(ts,ps);
    else
        if length(coord) + 2 != length(ncdim(nc{name}))
            error (sprintf('wrong number of dimensions given for %s', name));
        end
        if length(coord) == 1
            X = nc{name}(ts,coord(1),ps);
        elseif length(coord) == 2
            X = nc{name}(ts,coord(1),coord(2),ps);
        elseif length(coord) == 3
            X = nc{name}(ts,coord(1),coord(2),coord(3),ps);
        end
    end
    X = squeeze(X);
end
