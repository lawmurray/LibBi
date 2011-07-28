% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} surf_model (@var{model}, @var{mn}, @var{mx})
%
% Surface plot for 2-dimensional model.
%
% @itemize
% @bullet{ @var{model} Model, as output by model_*().}
%
% @bullet{ @var{ax} (optional) Axis range. If not specified, determined
% from model.
% @end deftypefn
%
function surf_model (model, ax)
    RES = 50;
    
    % check arguments
    if nargin < 1 || nargin > 2
        print_usage ();
    end
    if nargin < 2
        ax = [];
    end
    if columns (model.X) != 2
        error ('surf_likelihood only for 2d input');
    end
    
    allx = model.X(:,1);
    ally = model.X(:,2);
    
    if isempty (ax)
        ax1 = [ min(allx), max(allx), min(ally), max(ally) ];
        ax(1:2) = unstandardise(ax1(1:2)', model.mu(1), model.sigma(1));
        ax(3:4) = unstandardise(ax1(3:4)', model.mu(2), model.sigma(2));
    else
        ax1(1:2) = standardise(ax(1:2)', model.mu(1), model.sigma(1));
        ax1(3:4) = standardise(ax(3:4)', model.mu(2), model.sigma(2));
    end
    
    % determine kriging extents
    x = linspace(ax1(1), ax1(2), RES);
    y = linspace(ax1(3), ax1(4), RES);
    
    [XX YY] = meshgrid(x, y);
    Z = [ XX(:) YY(:) ];
    
    if isfield(model, 'type') && strcmp(model.type, 'posterior')
        % kde surface
        m = kernel_density(Z, model.X, model.h);
    else
        % krig surface
        [m s2] = gp(model.hyp, @infExact, model.meanfunc, model.covfunc, ...
            model.likfunc, model.X, model.y, Z);
        m = exp(m);
    end

    % determine visualisation extents
    x = linspace(ax(1), ax(2), RES);
    y = linspace(ax(3), ax(4), RES);   
    [XX YY] = meshgrid(x, y);
    ZZ = reshape(m, size(XX));
    
    % surface
    surf(XX, YY, ZZ);
end
