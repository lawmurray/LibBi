/**
 *
 */
def Gaussian(mu:Real, sigma:Real) {
  var mu:Real <- mu;
  var sigma:Real <- sigma;
}

def sample(rng:RNG, x:Real ~ p:Gaussian) {
  x <- p.sample(rng);
}

def ldensity(x:Real ~ p:Gaussian) -> l:Real {
  //
}
