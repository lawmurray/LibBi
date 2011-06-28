% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} surf_likelihood (@var{model}, @var{mn}, @var{mx})
%
% Surface plot for 2-dimensional likeilhood noise.
%
% @itemize
% @bullet{ @var{model} Model, as output by krig_likelihood().}
% @end deftypefn
%
function surf_likelihood (model)
    RES = 50;
    
    if nargin < 1 || nargin > 3
        print_usage ();
    end
    if columns (model.X) != 2
        error ('surf_likelihood only for 2d input');
    end
    
    allx = model.X(:,1);
    ally = model.X(:,2);
    
    x = linspace(min(allx), max(allx), RES);
    y = linspace(min(ally), max(ally), RES);
    
    [XX YY] = meshgrid(x, y);
    Z = [ XX(:) YY(:) ];
    
    [m s2] = gp(model.hyp, @infExact, model.meanfunc, model.covfunc, ...
        model.likfunc, model.X, model.logalpha, Z);

    ZZ = reshape(m, size(XX));
    
    % surface
    surf(XX, YY, exp(ZZ));
end
