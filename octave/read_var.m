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
    numdims = ncnumdims (nc, name);
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
    if numdims == 0
        X = nc{name}(:);
    elseif numdims == 1
        X = nc{name}(ps);
    else
        if length(coord) == 0
            if numdims == 1
                X = nc{name}(ps);
            else
                X = nc{name}(ts,ps);
            end            
        elseif length(coord) == 1
            if numdims == 2
                X = nc{name}(coord(1),ps);
            else
                X = nc{name}(ts,coord(1),ps);
            end
        elseif length(coord) == 2
            if numdims == 3
                X = nc{name}(coord(1),coord(2),ps);
            else
                X = nc{name}(ts,coord(1),coord(2),ps);
            end
        elseif length(coord) == 3
            if numdims == 4
                X = nc{name}(ts,coord(1),coord(2),coord(3),ps);
            else
                X = nc{name}(coord(1),coord(2),coord(3),ps);
            end
        end
    end
    X = squeeze(X);
end
