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
% @end itemize
% @end deftypefn
%
function model = krig_likelihood (model)
    meanfunc = @meanConst; hyp.mean = mean(model.logalpha);
    covfunc = @covSEiso; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
    likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);
    
    hyp = minimize(hyp, @gp, -500, @infExact, meanfunc, covfunc, likfunc, ...
                   model.X, model.logalpha);

    % result structure
    model.hyp = hyp;
    model.meanfunc = meanfunc;
    model.covfunc = covfunc;
    model.likfunc = likfunc;
end
