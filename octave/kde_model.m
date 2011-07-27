% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1808 $
% $Date: 2011-07-26 15:15:55 +0800 (Tue, 26 Jul 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} {@var{model} = } kde_model (@var{model})
%
% Fit kernel density estimate to model.
%
% @itemize
% @bullet{ @var{model} Model.}
% @end itemize
% @end deftypefn
%
function model = kde_model (model)
    if nargin != 1
        print_usage ();
    end

    model.h = kernel_optimal_bandwidth(model.X);
end
