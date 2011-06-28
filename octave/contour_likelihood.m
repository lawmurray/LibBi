% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} contour_likelihood (@var{model}, @var{mn}, @var{mx}, @var{ax}, @var{lvl})
%
% Surface plot for 2-dimensional likeilhood noise.
%
% @itemize
% @bullet{ @var{model} Model, as output by krig_likelihood().}
%
% @bullet{ @var{mn} (optional) Local minima, as output by
% minmax_likelihood().}
%
% @bullet{ @var{mx} (optional) Global maxima, as output by
% minmax_likelihood().}
%
% @bullet{ @var{ax} (optional) Axis range. If not specified, determined
% from model.
%
% @bullet{ @var{lvl} (optional) Contour levels.
% @end deftypefn
%
function contour_likelihood (model, mn, mx, ax, lvl)
    RES = 50;
    
    if nargin < 1 || nargin > 5
        print_usage ();
    end
    if nargin < 2
        mn = [];
    end
    if nargin < 3
        mx = [];
    end
    if nargin < 4
        ax = [];
    elseif length (ax) != 4
        error ('ax must be vector of length 4');
    end
    if nargin < 5
        lvl = [];
    end
    if columns (model.X) != 2
        error ('contour_likelihood only for 2d input');
    end
    
    % determine kriging extents
    if isempty (ax)
        allx = model.X(:,1);
        ally = model.X(:,2);
        if nargin > 1
            allx = [ allx; mn(:,1) ];
            ally = [ ally; mn(:,2) ];
        end
        if nargin > 2
            allx = [ allx; mx(:,1) ];
            ally = [ ally; mx(:,2) ];
        end
        ax1 = [ min(allx), max(allx), min(ally), max(ally) ];
        ax(1:2) = unstandardise(ax1(1:2)', model.mu(1), model.sigma(1));
        ax(3:4) = unstandardise(ax1(3:4)', model.mu(2), model.sigma(2));
    else
        ax1(1:2) = standardise(ax(1:2)', model.mu(1), model.sigma(1));
        ax1(3:4) = standardise(ax(3:4)', model.mu(2), model.sigma(2));
    end
    
    x = linspace(ax1(1), ax1(2), RES);
    y = linspace(ax1(3), ax1(4), RES);   
    [XX YY] = meshgrid(x, y);
    Z = [ XX(:) YY(:) ];
    
    % krig surface
    [m s2] = gp(model.hyp, @infExact, model.meanfunc, model.covfunc, ...
        model.likfunc, model.X, model.logalpha, Z);
    
    % determine visualisation extents
    x = linspace(ax(1), ax(2), RES);
    y = linspace(ax(3), ax(4), RES);   
    [XX YY] = meshgrid(x, y);
    ZZ = reshape(m, size(XX));

    % contour plot
    if isempty(lvl)
        contourf(XX, YY, exp(ZZ));
    else
        contourf(XX, YY, exp(ZZ), lvl);
    end
    
    % extrema
    if !isempty (mn)
        hold on;
        plot(mn(:,1), mn(:,2), '.w', 'markersize', 20);
    end
    if !isempty (mx)
        hold on;
        plot(mx(:,1), mx(:,2), '.w', 'markersize', 20);
    end
    
    % transects
    if !isempty (mn) && !isempty (mx)
       for i = 1:rows(mn)
           line([ mx(1,1) mn(i,1) ], [ mx(1,2) mn(i,2) ], 'color', 'w', ...
                'linewidth', 3);
       end
    end
    
    hold off;
end
