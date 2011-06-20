% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1603 $
% $Date: 2011-06-07 11:40:59 +0800 (Tue, 07 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} [@var{mn} @var{mx}] = minmax_likelihood (@var{model})
%
% Find global minimum (constrained to convex hull of support) and local maxima
% of likelihood noise surface.
%
% @itemize
% @bullet{ @var{model} Model, as output by krig_likelihood().}
% @end itemize
% @end deftypefn
%
function [mn mx] = minmax_likelihood (model)
    N = 20;
    options = zeros (10,1);
    options(6) = 1;
    options(10) = 500;
    
    % maxima
    mx = model.X(randperm (rows (model.X))(1:N),:);
    for i = 1:N
        mx(i,:) = fmins('maxgp', mx(i,:), options, [], model.hyp, ...
            @infExact, model.meanfunc, model.covfunc, model.likfunc, ...
            model.X, model.sigma);    
    end

    % constrained minima
    %H = convhulln (model.X);
    mn = model.X(randperm (rows (model.X))(1:N),:);
    for i = 1:N
        mn(i,:) = fmins('mingp', mn(i,:), options, [], model.hyp, ...
            @infExact, model.meanfunc, model.covfunc, model.likfunc, ...
            model.X, model.sigma);
    end

    % pick global minima
    vals = gp(model.hyp, @infExact, model.meanfunc, model.covfunc, ...
        model.likfunc, model.X, model.sigma, mn);
    [val i] = min(vals);
    mn = mn(i,:);

    save minmax.mat mn mx
    
    % eliminate duplicate maxima (using 5 sig figs for comparison)
    mx = unique(map(@trim, mx), 'rows');
end
