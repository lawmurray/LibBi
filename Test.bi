model Test {
  param theta, sigma2;
  noise w;
  state x;

  sub parameter {
    theta ~ gaussian();
    sigma2 ~ inverse_gamma();
  }

  sub initial {
    x ~ gaussian();
  }

  sub transition {
    w ~ gaussian(0.0, sqrt(sigma2));
    x <- theta*x + w;
  }
}
