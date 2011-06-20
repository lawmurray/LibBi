% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1603 $
% $Date: 2011-06-07 11:40:59 +0800 (Tue, 07 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} krig_likelihood (@var{in}, @var{invars}, @var{coords}, @var{M})
%
% Krig likelihood noise.
%
% @itemize
% @bullet{ @var{in} Input file. Gives the name of a NetCDF file output by
% pf.}
%
% @bullet{ @var{invars} Name of variables from input file to include.
%
% @bullet{ @var{coords} Cell array giving vectors of spatial
% coordinates of zero to three elements, giving the x, y and z coordinates of
% the corresponding variable in @var{invars} to plot.}
%
% @bullet{ @var{M} Number of repeated likelihood computations for each
% sample in file.
% @end itemize
% @end deftypefn
%
function model = krig_likelihood (in, invars, coords, M)
    % read in log-likelihoods
    nc = netcdf('results/likelihood_disturbance.nc.0', 'r');
    ll = nc{'loglikelihood'}(:);
    ll = reshape(ll, M, length(ll)/M);

    % compute log of mean likelihood
    mx = max(ll);
    llZ = ll - repmat(mx, rows(ll), 1);
    lZ = exp(llZ);
    logmu = log(mean(lZ)) + mx;
    
    % compute standard deviation of likelihood
    ll0 = ll - repmat(logmu, rows(ll), 1);
    l0 = exp(ll0);
    sigma = std(l0);
    
    logmu = logmu';
    sigma = sigma';
    
    % support points
    X = [];
    for i = 1:length(invars)
        X = [ X, nc{invars{i}}(1:M:end) ];
    end

    % krig likelihood noise
    meanfunc = @meanZero;
    covfunc = @covSEiso; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
    likfunc = @likGauss; sn = 1.0; hyp.lik = log(sn);
    
    hyp = minimize(hyp, @gp, -200, @infExact, meanfunc, covfunc, likfunc, ...
                   X, sigma);

    % result structure
    model.hyp = hyp;
    model.meanfunc = meanfunc;
    model.covfunc = covfunc;
    model.likfunc = likfunc;
    model.X = X;
    model.logmu = logmu;
    model.sigma = sigma;
end
