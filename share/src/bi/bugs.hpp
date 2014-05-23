/**
 * @bug sample.nc (output of bi sample) has a time variable but it is blank
 * NOTE: we can get the time from filter.nc but it would be good to have it in here.
 *
 * @bug smc^2 output has an additional time slot at the end that is blank for
 * all variables.
 *
 * @bug smc^2 output has variables for states (e.g. alpha, P, Z) but they are
 * blank (fixed by removing D_VARS and R_VARS from MarginalSIR output).
 *
 * @bug ANPF filter.nc output has some garbage values (unaffected by recent
 * refactorization).
 *
 * @bug simulate and filter put P parameter values into the results file
 * instead of one.
 *
 * @bug --enable-extradebug produces compile errors when used in conjunction
 * with --enable-cuda. Filtered the --fno-inline flag in nvcc_wrapper.pl,
 * but hasn't fixed everything.
 */
