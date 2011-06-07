% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev: 1371 $
% $Date: 2011-04-04 13:51:32 +0800 (Mon, 04 Apr 2011) $

% -*- texinfo -*-
% @deftypefn {Function File} {@var{nice} = } nice_greek (@var{name})
%
% Convert named Greek letter to Greek symbol. If not a named Greek letter,
% returns @var{name}.
%
% @itemize
% @bullet{ @var{name} Name of the variable.}
% @end itemize
% @end deftypefn
%
function nice = nice_greek (name)
    switch name
      case {'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'varepsilon', ...
            'zeta', 'eta', 'theta', 'vartheta', 'iota', 'kappa', 'lambda', ...
            'mu', 'nu', 'xi', 'pi', 'varpi', 'rho', 'varrho', 'sigma', ...
            'varsigma', 'tau', 'upsilon', 'phi', 'varphi', 'chi', 'psi', ...
            'omega', 'Gamma', 'Delta', 'Theta', 'Lambda', 'Xi', 'Pi', ...
            'Sigma', 'Upsilon', 'Phi', 'Psi', 'Omega'}
        nice = strcat('\', name);
      otherwise
        nice = name;
    end
end

