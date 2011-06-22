function d = dmingp (z, model)
    c = model.covfunc(model.hyp.cov, model.X, z);
    
    ell2 = exp(2.0*model.hyp.cov(1));
    C = repmat(c, 1, columns(model.X));
    Z = repmat(z, rows(model.X), 1);
    Y = -C.*(Z - model.X)/ell2;
    d = model.k'*Y;
end
