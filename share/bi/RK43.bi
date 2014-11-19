/**
 * Numerical integration of ODE system using RK4(3) method.
 */
method eval(ODE (args) { statements }) -> {
  /**
   * State of ODE system.
   */
  class ODEState {
    map(statements, transform { d(x[coords])/d(t) = expr } -> { var x[dims]:Real });
  
    /**
     * Evaluate derivatives.
     */
    method d(dx:ODEState) {
      map(statements, transform { d(x[coords])/d(t) = expr } -> { dx.x[coords] <- expr });
    }
  }

  /**
   * Assignment between model and state variables.
   */  
  method eval(o1 <- o2:ODEState) {
    map(statements, transform { d(x[coords])/d(t) = expr } -> { o1.x[coords] = o2.x[coords]; });
  }
  
  /**
   * Assignment between state and model variables.
   */  
  method eval(o1:ODEState <- o2) {
    map(statements, transform { d(x[coords])/d(t) = expr } -> { o1.x[coords] = o2.x[coords]; });
  }

  var r1:ODEState;
  var r2:ODEState;
  var err:ODEState;
  var old:ODEState;
  
  var t:Real;
  var h:Real;
  var e:Real;
  var e2:Real;
  var logfacold:Real;
  var logfac11:Real;
  var fac:Real;
  var n:Int;
  var id:Int;
  var p:Int;

  /* coefficients */
  const a21:Real = 0.225022458725713;
  const b1:Real = 0.0512293066403392;
  const e1:Real = -0.0859880154628801; // b1 - b1hat
  const a32:Real = 0.544043312951405;
  const b2:Real = 0.380954825726402;
  const c2:Real = 0.225022458725713;
  const e2:Real = 0.189074063397015; // b2 - b2hat
  const a43:Real = 0.144568243493995;
  const b3:Real = -0.373352596392383;
  const c3:Real = 0.595272619591744;
  const e3:Real = -0.144145875232852; // b3 - b3hat
  const a54:Real = 0.786664342198357;
  const b4:Real = 0.592501285026362;
  const c4:Real = 0.576752375860736;
  const e4:Real = -0.0317933915175331; // b4 - b4hat
  const b5:Real = 0.34866717899928;
  const c5:Real = 0.845495878172715;
  const e5:Real = 0.0728532188162504; // b5 - b5hat
    
  t <- t1;
  h <- h_h0;
  logfacold <- log(1.0e-4);
  n <- 0;
  old <- this;
  r1 <- old;
  
  while (t < t2 && n < h_nsteps) {
    if (0.1*abs(h) <= abs(t)*h_uround) {
      // step size too small
    }
    if (t + 1.01*h - t2 > 0.0) {
      h <- t2 - t;
    }
    if (h <= 0.0) {
      t <- t2;
    } else {
      /* stages */
      r1.d(r2);
      r2 <- a*r1;
      //...etc
            
      /* compute error */
      e2 <- 0.0;
      for ([i=1:N]) {
        e <- err[i]*h/(h_atoler + h_rtoler*max(abs(old[i]), abs(r1[i])));
        e2 <- e2 + e*e;
      }
      e2 <- e2/N;
      
      if (es <= 1.0) {
        /* accept */
        t <- t + h;
        if (t < t2) {
          old <- r1;
        }
      } else {
        /* reject */
        r1 <- old;
        this <- old;
      }
      
      /* next step size */
      if (t < t2) {
        logfac11 <- h_expo*log(e2);
        if (e2 > 1.0) {
          /* step was rejected */
          h <- h*max(h_facl, exp(h_logsafe - logfac11));
        } else {
          /* step was accepted */
          fac <- exp(h_beta*logfacold + h_logsafe - logfac11);  // Lund-stabilization
          fac <- min(h_facr, max(h_facl, fac));  // bound
          h <- h*fac;
          logfacold = 0.5*log(max(e2, 1.0e-8));
        }
      }
      
      n <- n + 1;
    }
  }
}
