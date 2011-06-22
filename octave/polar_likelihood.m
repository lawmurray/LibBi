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
% @bullet{ @var{model} Model, as output by model_likelihood().}
%
% @bullet{ @var{mn} (optional) Global minima, as output by
% minmax_likelihood().}
%
% @bullet{ @var{mx} (optional) Local maxima, as output by
% minmax_likelihood().}
% @end deftypefn
%
function polar_likelihood (model, mn, mx, ncontours)
    RES_THETA = 24;
    ANGLE = 2*pi/(rows(mn)*1.5);
    
    if nargin < 3 || nargin > 4
        print_usage ();
    end
    if nargin < 4
        ncontours = 20;
    end
    ncontours += 1;
           
    % compute rays
    Z1 = kron(mn, ones(ncontours,1));
    Z2 = repmat(mx(1,:), ncontours*rows(mn), 1);
    alpha = repmat([0:(1.0/(ncontours-1)):1.0]', rows(mn), columns(mn));
    Z = alpha.*Z1 + (1.0 - alpha).*Z2;
    
    [m s2] = gp(model.hyp, @infExact, model.meanfunc, model.covfunc, ...
        model.likfunc, model.X, model.logalpha, Z);
    
    d = sqrt(sumsq(Z - Z2, 2)); % distances

    m = reshape(m, ncontours, rows(mn));
    d = reshape(d, ncontours, rows(mn));
    caxis([min(m(:)), max(m(:))]);
    %axis([-max(d(:)), max(d(:)), -max(d(:)), max(d(:))]);
    
    % setup
    clf;
    polar(linspace(0, 2*pi, 360), repmat(max(d(:)), 1, 360), '-k');

    % plot rays
    thetares = linspace(0, ANGLE, RES_THETA)';
    for i = 1:columns(d)
        theta = kron(i*2*pi/rows(mn) - ANGLE/2 + thetares, ones(ncontours,1));
        rho = repmat(d(:,i), length(thetares), 1);
        pol = [ theta, rho ];
        cart = pol2cart(pol);
        
        V = cart;
        F = repmat([1:ncontours-1]', 1, length(thetares));
        inc = repmat(ncontours*[0:columns(F) - 1], rows(F), 1);
        F += inc;
        F = [ F, fliplr(F + 1)];
        C = m(2:end,i);
        
        patch('Faces', F, 'Vertices', V, 'FaceVertexCData', C, 'LineWidth', ...
              0, 'FaceAlpha', 0.7);
    end
        
    % tidy up
    hold off;
    box off;
    %axis("tight");
    %ax = max(abs(axis()));
    %axis([-ax ax -ax ax], 'square', 'off');      
end
