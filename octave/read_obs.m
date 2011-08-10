% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} {@var{y} = } read_obs (@var{nc}, @var{name}, @var{coord}, @var{ts}, @var{ns})
%
% Read observation from NetCDF file.
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
% @bullet{ @var{ts} (optional) Time indices.
%
% @bullet{ @var{ns} (optional) Index along ns dimension of input file.}
% @end itemize
% @end deftypefn
%
function [t y] = read_obs (nc, name, coord, ts, ns)
    % check arguments
    if nargin < 2 || nargin > 5
        print_usage ();
    end
    if nargin < 3
        coord = [];
    end
    if nargin < 4
        ts = []
    end
    if nargin < 5
        ns = 1;
    end
    
    if length (coord) > 3
        error ('coord must be vector of length zero to three');
    end
    
    % check number of dimensions
    numdims = length (ncdim (nc{name}));
    if ncdimexists (nc, sprintf('nr_%s', name))
        T = length (nc(sprintf('nr_%s', name)));
    elseif ncdimexists (nc, 'nr')
        T = length (nc('nr'));    
    else
        T = 1;
    end
    if isempty (ts)
        ts = [1:T];
    end
    
    % time variable
    tvar = sprintf('time_%s', name);
    if !ncvarexists(nc, tvar)
        tvar = 'time';
    end
    K = length (nc{tvar}(:));
    
    % coordinate variable
    dense = 0;
    cvar = sprintf('coord_%s', name);
    if !ncvarexists(nc, cvar)
        cvar = 'coord';
        if !ncvarexists(nc, cvar)
            cvar = '';
            dense = 1;
        end
    end
    if !dense
        if length(coord) > 1
            numdims = length (ncdim (nc{name}));
            if numdims == 1
                coords = nc{cvar}(:);
            elseif numdims == 2
                coords = nc{cvar}(:,:);
            else
                error (sprintf('Variable %s has too many dimensions', cvar));
            end
            coords = coords + 1; % offset from base 0 to base 1 indexing
        else
            coords = [];
        end
    end

    % read
    if dense
        switch length(coord)
          case 0
            ys = nc{name}(ns,ts);
          case 1
            ys = nc{name}(ns,ts,coord(1));
          case 2
            ys = nc{name}(ns,ts,coord(1),coord(2));
          case 3
            ys = nc{name}(ns,ts,coord(1),coord(2),coord(3));
        end
        t = nc{tvar}(ts);
        y = ys;
    else
        ys = nc{name}(ns,:);

        % mask based on coordinates
        if isempty(coords)
            mask = ones(K,1);
        else
            mask = zeros(K,1);
            for k = 1:K
                if coords(k,:) == coord
                    mask(k) = 1;
                end
            end
        end
        t = nc{tvar}(find(mask));
        y = ys(find(mask));
    end
end
