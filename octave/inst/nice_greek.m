% Copyright (C) 2011
% Author: Lawrence Murray <lawrence.murray@csiro.au>
% $Rev$
% $Date$

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
    greek = {'alpha'; 'beta'; 'gamma'; 'delta'; 'epsilon'; 'varepsilon'; ...
            'zeta'; 'eta'; 'theta'; 'vartheta'; 'iota'; 'kappa'; 'lambda'; ...
            'mu'; 'nu'; 'xi'; 'pi'; 'varpi'; 'rho'; 'varrho'; 'sigma'; ...
            'varsigma'; 'tau'; 'upsilon'; 'phi'; 'varphi'; 'chi'; 'psi'; ...
            'omega'; 'Gamma'; 'Delta'; 'Theta'; 'Lambda'; 'Xi'; 'Pi'; ...
            'Sigma'; 'Upsilon'; 'Phi'; 'Psi'; 'Omega'};
    nice = name;
    for i = 1:length(greek)
        letter = greek{i};
        if strfind (name, letter) == 1
            nice = strcat('{\', letter);
            if length(name) > length(letter)
                nice = strcat(nice, '_{', name(length(letter) + 1:end), ...
                          '}');
            end
            nice = strcat(nice, '}');
            break;
        end
    end
end

