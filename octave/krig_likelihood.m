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
    nc = netcdf(in, 'r');
    P = length(nc('np'));
    
    % read in log-likelihoods
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
    
    % bootstrap noise in sigma to get gp signal noise
    sns = zeros(length(sigma), 1);
    for i = 1:length(sigma)
        u = randi(rows(l0), 1000*rows(l0), 1);
        l = reshape(l0(u,i), rows(l0), 1000);
        sigmas = std(l);
        sns(i) = std(sigmas(:));
    end
    
    % support points
    X = [];
    for i = 1:length(invars)
        if length(coords) >= i
            x = read_var(nc, invars{i}, coords{i}, [1:M:P], 1);
        else
            x = read_var(nc, invars{i}, [], [1:M:P], 1);
        end    
        X = [ X, x(:) ];
    end

    % krig likelihood noise
    sn = max(sns);
    meanfunc = @meanZero;
    covfunc = @covSEiso; ell = 1/4; sf = 1; hyp.cov = log([ell; sf]);
    likfunc = @likGauss; hyp.lik = log(sn);
    
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
