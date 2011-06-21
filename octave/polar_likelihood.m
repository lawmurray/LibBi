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
function polar_likelihood (model, mn, mx)
    RES_RHO = 11;
    RES_THETA = 24;
    ANGLE = 2*pi/(rows(mn)*1.5);
    
    if nargin != 3
        print_usage ();
    end
    
    % setup
    clf;
    polar(0,0);
    
    % plot distance circles
    d = sqrt(sumsq(mn - repmat(mx(1,:), rows(mn), 1), 2)); % distances
    [d, is] = sort(d);
    
    %theta = linspace(0, 2*pi, 360);
    %for i = 1:length(d)
    %    rho = repmat(d(i), length(theta), 1);
    %    polar(theta, rho, ':k');
    %    hold on
    %end
    
    % compute rays
    Z1 = kron(mn, ones(RES_RHO,1));
    Z2 = repmat(mx(1,:), RES_RHO*rows(mn), 1);
    alpha = repmat([0:(1.0/(RES_RHO-1)):1.0]', rows(mn), columns(mn));
    Z = alpha.*Z1 + (1.0 - alpha).*Z2;
    
    [m s2] = gp(model.hyp, @infExact, model.meanfunc, model.covfunc, ...
        model.likfunc, model.X, model.logalpha, Z);
    
    d = sqrt(sumsq(Z - Z2, 2)); % distances

    m = reshape(m, RES_RHO, rows(mn));
    d = reshape(d, RES_RHO, rows(mn));
    caxis(exp([min(m(:)), max(m(:))]));
    
    % plot rays
    thetares = linspace(0, ANGLE, RES_THETA)';
    for j = 1:length(is)
        i = is(j);
        theta = kron(i*2*pi/rows(mn) - ANGLE/2 + thetares, ones(RES_RHO,1));
        rho = repmat(d(:,i), length(thetares), 1);
        pol = [ theta, rho ];
        cart = pol2cart(pol);
        
        V = cart;
        F = repmat([1:RES_RHO-1]', 1, length(thetares));
        inc = repmat(RES_RHO*[0:columns(F) - 1], rows(F), 1);
        F += inc;
        F = [ F, fliplr(F + 1)];
        C = exp(m(2:end,i));
        
        patch('Faces', F, 'Vertices', V, 'FaceVertexCData', C);
    end
        
    % tidy up
    hold off;
    axis("tight");
    ax = max(abs(axis()));
    axis([-ax ax -ax ax], 'square', 'off');      
end
