% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1603 $
% $Date: 2011-06-07 11:40:59 +0800 (Tue, 07 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} polar_likelihood (@var{model}, @var{mn}, @var{mx})
%
% Polar plot for likelihood noise.
%
% @itemize
% @bullet{ @var{model} Model, as output by krig_likelihood().}
%
% @bullet{ @var{mn} (optional) Global minima, as output by
% minmax_likelihood().}
%
% @bullet{ @var{mx} (optional) Local maxima, as output by
% minmax_likelihood().}
% @end deftypefn
%
function polar_likelihood (model, mn, mx)
    RES_RHO = 21;
    RES_THETA = 24;
    
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

    Z1 = kron(mx, ones(RES_RHO,1));
    Z2 = repmat(mn(1,:), RES_RHO*rows(mx), 1);
    alpha = repmat([0:(1.0/(RES_RHO-1)):1.0]', rows(mx), columns(mx));
    Z = alpha.*Z1 + (1.0 - alpha).*Z2;
    
    [m s2] = gp(model.hyp, @infExact, model.meanfunc, model.covfunc, ...
        model.likfunc, model.X, model.sigma, Z);
    
    d = sqrt(sumsq(Z - Z2, 2)); % distances

    m = reshape(m, RES_RHO, rows(mx));
    d = reshape(d, RES_RHO, rows(mx));
    Mx = max(m(:));
    
    clf;
    polar(0,0);
    thetares = linspace(0, 2*pi/12.0, RES_THETA)';
    for i = 1:rows(mx)    
        theta = kron(i*2*pi/rows(mx) + thetares, ones(RES_RHO,1));
        rho = repmat(d(:,i), length(thetares), 1);
        pol = [ theta, rho ];
        cart = pol2cart(pol);
        
        V = cart;
        F = repmat([1:RES_RHO-1]', 1, length(thetares));
        inc = repmat(RES_RHO*[0:columns(F) - 1], rows(F), 1);
        F += inc;
        F = [ F, fliplr(F + 1)];
        C = floor(m(2:end,i)/Mx*length(colormap()));
        
        patch('Faces', F, 'Vertices', V, 'FaceVertexCData', C, 'EdgeColor', ...
              'k');
    end
    hold off;
    axis("tight");
    ax = max(abs(axis()));
    axis([-ax ax -ax ax], 'square', 'off');      
end
