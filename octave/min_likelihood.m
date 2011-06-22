% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1603 $
% $Date: 2011-06-07 11:40:59 +0800 (Tue, 07 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} @var{mn}= min_likelihood (@var{model})
%
% Find local minima of acceptance rate surface.
%
% @itemize
% @bullet{ @var{model} Model, as output by krig_likelihood().}
%
% @bullet{ @var{attempts} Number of optimisation runs to attempt.
%
% @bullet{ @var{maxiters} Maximum number of iterations in each optimisation
% run.
% @end itemize
% @end deftypefn
%
function mn = min_likelihood (model, attempts, maxiters)
    if nargin < 1 || nargin > 3
        print_usage ();
    end
    if nargin < 2
        attempts = ceil(sqrt(rows(model.X)));
    end
    if nargin < 3
        maxiters = 500;
    end

    % optimisation options
    options = zeros (10,1);
    options(10) = maxiters;
    
    % minima
    mn = model.X(randperm (rows (model.X))(1:attempts),:);
    for i = 1:attempts
        mn(i,:) = fmins('mingp', mn(i,:), options, [], model.hyp, ...
            @infExact, model.meanfunc, model.covfunc, model.likfunc, ...
            model.X, model.logalpha);    
    end

    % eliminate duplicate minima (using 5 sig figs for comparison)
    mn = unique(map(@trim, mn), 'rows');
    
    % eliminate any minima that are really just mean of Gaussian process
    vals = gp(model.hyp, @infExact, model.meanfunc, model.covfunc, ...
        model.likfunc, model.X, model.logalpha, mn);
    is = find(vals != model.hyp.mean);
    mn = mn(is,:);
end
