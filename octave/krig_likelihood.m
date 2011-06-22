% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1603 $
% $Date: 2011-06-07 11:40:59 +0800 (Tue, 07 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} krig_likelihood (@var{model})
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
function model = krig_likelihood (model, maxiters)
    if nargin < 1 || nargin > 2
        print_usage ();
    end
    if nargin < 2
        maxiters = 200;
    end
    
    meanfunc = @meanConst; hyp.mean = mean(model.logalpha);
    covfunc = @covSEiso; ell = 10; sf = 1; hyp.cov = log([ell; sf]);
    likfunc = @likGauss; sn = 5.0; hyp.lik = log(sn);
    
    hyp = minimize(hyp, @gp, -maxiters, @infExact, meanfunc, covfunc, ...
        likfunc, model.X, model.logalpha);
    
    % result structure
    model.hyp = hyp;
    model.meanfunc = meanfunc;
    model.covfunc = covfunc;
    model.likfunc = likfunc;
    
    % precomputes for later
    K = covfunc(model.hyp.cov, model.X);
    model.k = cholinv(K)'*(model.logalpha - model.hyp.mean);
end
