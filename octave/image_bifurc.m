% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1551 $
% $Date: 2011-05-24 13:50:51 +0800 (Tue, 24 May 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} image_bifurc (@var{in}, @var{xvar}, @var{xcoord}, @var{yvar}, @var{ycoords}, @var{rang})
%
% Plot bifurcation image using output of simulate program.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% simulate.}
%
% @bullet{ @var{xvar} Name of variable from input file to use along x-axis
% of plot. Typically this is a parameter of the model.
%
% @bullet{ @var{xcoord} Vector of spatial coordinates of zero
% to three elements, giving the x, y and z coordinates of a
% component of @var{xvar} to use.}
%
% @bullet{ @var{yvar} Name of variable from input file to use along y-axis
% of plot. Typically this is a dynamic state variable of the model.
%
% @bullet{ @var{xcoords} (optional) Matrix of spatial coordinates of zero
% to three columns, giving the x, y and z coordinates of the components of
% @var{yvar} to use. These will all be combined in producing the
% plot.}
%
% @bullet{ @var{rang} (optional) Vector of indices of times to
% include. Useful for excluding burn-in periods, for instance.
% @end itemize
% @end deftypefn
%
function image_bifurc (in, xvar, xcoord, yvar, ycoords, rang)    
    % constants
    RES_X = 1200;
    RES_Y = 800;
        
    % check arguments
    if nargin < 4 || nargin > 6
        print_usage ();
    end
    if nargin < 5
        ycoords = [];
        rang = [];
    elseif nargin < 6
        rang = [];
    end
    if length (xcoord) > 3
        error ('xcoord should be a vector with at most three elements');
    end
    if !(ismatrix (ycoords) && columns (ycoords) <= 3)
        error ('ycoords should be a matrix with at most three columns');
    end
    
    % input file
    nci = netcdf(in, 'r');

    % data
    P = length (nci('np'));
    T = length (nci('nr'));
    if length (rang) == 0
        rang = [1:T];
    end
    K = rows(ycoords);
    if K == 0
        K = 1;
    end

    x = read_var (nci, xvar, xcoord, [1:P], 1)';
    Y = zeros(length(x), K*length(rang));
    for i = 1:rows(ycoords)
        y = read_var (nci, yvar, ycoords(i,:), [1:P], rang)';
        Y(:,(i - 1)*length(rang)+1:i*length(rang)) = y;
    end
    ncclose(nci);
    
    % data extents
    xmin = min(x);
    xmax = max(x);
    ymin = min(Y(:));
    ymax = max(Y(:));
    xs = linspace(xmin, xmax, RES_X);
    ys = linspace(ymin, ymax, RES_Y);
    
    % histogram
    n1 = histc(Y, ys, 2);
    n2 = zeros(RES_X, RES_Y);
    
    [n3,ii] = histc(x, xs);
    n2 = accumdim(ii, n1, 1);
    
    % plot
    n2 = n2' ./ repmat(max(n2'), rows(n2'), 1);
    imagesc(xs, ys, n2);
    axis([xmin xmax ymin ymax]);
end
