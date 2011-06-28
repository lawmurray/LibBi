% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1687 $
% $Date: 2011-06-28 11:46:45 +0800 (Tue, 28 Jun 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} standardise (@var{X}, @var{mu}, @var{sigma})
%
% Standardise rows of @var{X} using given mean and standard deviation
% vectors.
%
% @itemize
% @bullet{ @var{X}}
%
% @bullet{ @var{mu} Mean vector.}
%
% @bullet{ @var{sigma} Standard deviation vector.}
% @end deftypefn
%
function Z = standardise (X, mu, sigma)
    % check arguments
    if nargin != 3
        print_usage ();
    end
    if !isrow (mu)
        error ('mu should be row vector');
    end
    if !isrow (sigma)
        error ('sigma should be row vector');
    end
    if columns (X) > 0 
        if columns (X) != columns (mu)
            error ('mu should have same number of columns as Z');
        end
        if columns (X) != columns (sigma)
            error ('sigma should have same number of columns as Z');
        end
        
        Mu = repmat(mu, rows(X), 1);
        Sigma = repmat(sigma, rows(X), 1);
        Z = (X - Mu)./Sigma;
    else
        Z = [];
    end
end
