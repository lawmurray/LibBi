/**
 *
 */
model Gaussian(mu:Real, sigma:Real) {
  var mu:Real <- mu;
  var sigma:Real <- sigma;
}

function sample(rng:RNG, x:Real ~ p:Gaussian) {
  x <- p.sample(rng);
}

function ldensity(x:Real ~ p:Gaussian) -> l:Real {
  //
}
