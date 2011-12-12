% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

% -*- texinfo -*-
% @deftypefn {Function File} {@var{model} = } krig_model (@var{model}, @var{maxiters})
%
% Krig likelihood noise.
%
% @itemize
% @bullet{ @var{model} Model.}
%
% @bullet{ @var{maxiters} Maximum number of iterations in each optimisation
% run.
% @end itemize
% @end deftypefn
%
function model = krig_model (model, maxiters)
    if nargin < 1 || nargin > 2
        print_usage ();
    end
    if nargin < 2
        maxiters = 200;
    end
    
    meanfunc = 'meanConst'; hyp.mean = mean(model.y);
    covfunc = 'covSEiso'; ell = quantile(pdist(model.X), 0.05, 2); sf = 1; hyp.cov = log([ell; sf]);
    likfunc = 'likGauss'; sn = std(model.y); hyp.lik = log(sn);
    
    hyp = minimize(hyp, @gp, -maxiters, @infExact, meanfunc, covfunc, ...
        likfunc, model.X, model.y);
    
    % result structure
    model.hyp = hyp;
    model.meanfunc = meanfunc;
    model.covfunc = covfunc;
    model.likfunc = likfunc;
    
    % precomputes for later
    sn = exp(2.0*model.hyp.lik);
    K = feval(covfunc, model.hyp.cov, model.X);
    K = K + sn*eye(rows(K));
    model.k = K\(model.y - model.hyp.mean);
end
