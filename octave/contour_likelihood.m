% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1603 $
% $Date: 2011-06-07 11:40:59 +0800 (Tue, 07 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} contour_likelihood (@var{model}, @var{mn}, @var{mx})
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
% @end deftypefn
%
function contour_likelihood (model, mn, mx)
    RES = 50;
    
    if nargin < 1 || nargin > 3
        print_usage ();
    end
    if nargin < 2
        mn = [];
    end
    if nargin < 3
        mx = [];
    end
    if columns (model.X) != 2
        error ('surf_likelihood only for 2d input');
    end
    
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
    
    x = linspace(min(allx), max(allx), RES);
    y = linspace(min(ally), max(ally), RES);
    
    [XX YY] = meshgrid(x, y);
    Z = [ XX(:) YY(:) ];
    
    [m s2] = gp(model.hyp, @infExact, model.meanfunc, model.covfunc, ...
        model.likfunc, model.X, model.logalpha, Z);

    ZZ = reshape(m, size(XX));
    
    % contours
    contourf(XX, YY, exp(ZZ));
    
    % extreme
    if nargin > 1
        hold on;
        plot(mn(:,1), mn(:,2), '.w', 'markersize', 20);
    end
    if nargin > 2
        hold on;
        plot(mx(:,1), mx(:,2), '.w', 'markersize', 20);
    end
    
    % transects
    if nargin > 2
       for i = 1:rows(mn)
           line([ mx(1,1) mn(i,1) ], [ mx(1,2) mn(i,2) ], 'color', 'w', ...
                'linewidth', 3);
       end
    end
    
    hold off;
end
