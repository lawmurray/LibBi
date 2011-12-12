% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} contour_model (@var{model}, @var{mn}, @var{mx}, @var{ax}, @var{lvl})
%
% Surface plot for 2-dimensional likeilhood noise.
%
% @itemize
% @bullet{ @var{model} Model, as output by krig_likelihood().}
%
% @bullet{ @var{mn} (optional) Local minima, as output by min_model().}
%
% @bullet{ @var{mx} (optional) Global maxima, as output by max_model().}
%
% @bullet{ @var{ax} (optional) Axis range. If not specified, determined
% from model.
%
% @bullet{ @var{lvl} (optional) Contour levels.
% @end deftypefn
%
function contour_model (model, mn, mx, ax, lvl)
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
        X1 = unstandardise(model.X, model.mu, model.Sigma);
        allx = X1(:,1);
        ally = X1(:,2);
        if nargin > 1
            allx = [ allx; mn(:,1) ];
            ally = [ ally; mn(:,2) ];
        end
        if nargin > 2
            allx = [ allx; mx(:,1) ];
            ally = [ ally; mx(:,2) ];
        end
        ax = [ min(allx), max(allx), min(ally), max(ally) ];
    end
    
    x = linspace(ax(1), ax(2), RES);
    y = linspace(ax(3), ax(4), RES);   
    [XX YY] = meshgrid(x, y);
    Z = [ XX(:) YY(:) ];
    Z1 = standardise(Z, model.mu, model.Sigma);
    
    if isfield(model, 'type') && strcmp(model.type, 'posterior')
        % kde surface
        m = kernel_density(Z1, model.X, model.h);
    else
        % krig surface
        [m s2] = gp(model.hyp, @infExact, model.meanfunc, model.covfunc, ...
        model.likfunc, model.X, model.y, Z1);
        m = exp(m);
    end
    
    % contour plot
    ZZ = reshape(m, size(XX));
    if isfield(model, 'type') && strcmp(model.type, 'posterior')
        % kde surface
        if isempty(lvl)
            [C, lvl] = contourc(XX, YY, ZZ, 3);
        end
        contour(XX, YY, ZZ, lvl, 'edgecolor', watercolour(2), 'linewidth', 2);
    else
        if isempty(lvl)
            contourf(XX, YY, ZZ);
        else
            contourf(XX, YY, ZZ, lvl);
        end
    end
    
    % extrema
    if !isempty (mn)
        hold on;
        plot(mn(:,1), mn(:,2), '.w', 'markersize', 15);
    end
    if !isempty (mx)
        hold on;
        plot(mx(:,1), mx(:,2), '.w', 'markersize', 15);
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
