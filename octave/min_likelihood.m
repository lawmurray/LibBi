% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

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
    options = zeros (18,1);
    options(2) = 1.0e-6;
    %options(9) = 1; % check gradient
    options(14) = maxiters;
    
    % minima
    if attempts <= rows (model.X)
        mn = model.X(randperm (rows (model.X))(1:attempts),:);
    else
        mn = model.X(randi (rows (model.X), attempts, 1),:);
    end
    for i = 1:attempts
        mn(i,:) = scg(@mingp, mn(i,:), options, @dmingp, model);
        %mn(i,:) = fmins(@mingp, mn(i,:), [], [], model);
    end

    % eliminate duplicates
    mn = unique(map(@trim, mn), 'rows');
    
    % eliminate any that are just mean of Gaussian process
    vals = mingp(mn, model);
    is = find(trim(vals) != trim(model.hyp.mean));
    mn = mn(is,:);
    
    % sort
    vals = mingp(mn, model);
    [vals is] = sort(vals);
    mn = mn(is,:);
    
    % pick global
    %vals = mingp(mn, model);
    %[val i] = min(vals);
    %mn = mn(i,:);
end
