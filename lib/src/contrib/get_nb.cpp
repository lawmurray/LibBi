/*
    -- MAGMA (version 0.2) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       November 2009
*/

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for spotrf based on n
*/ 
extern "C"
int magma_get_spotrf_nb(int n){
  if (n <= 3328)
    return 128;
  else if (n<=4256)
    return 224;
  else 
    return 288;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sgeqrf based on m
*/
extern "C"
int magma_get_sgeqrf_nb(int m){
  if (m <= 2048)
    return 32;
  else if (m<=4032)
    return 64;
  else
    return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sgetrf based on m;
      the return value should be multiple of 64
*/
extern "C"
int magma_get_sgetrf_nb(int m){
  if (m <= 2048)
    return 64;
  else
    return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for spotrf based on n
*/ 
extern "C"
int magma_get_dpotrf_nb(int n){
  if (n <= 3328)
    return 128;
  else if (n<=4256)
    return 128;
  else 
    return 256;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sgeqrf based on m
*/
extern "C"
int magma_get_dgeqrf_nb(int m){
  if (m <= 2048)
    return 64;
  else 
    return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sgeqlf based on m
*/
extern "C"
int magma_get_sgeqlf_nb(int m){
  if (m <= 1024)
    return 32;
  else if (m<=4032)
    return 64;
  else
    return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sgeqlf based on m
*/
extern "C"
int magma_get_sgelqf_nb(int m){
  if (m <= 2048)
    return 32;
  else if (m<=4032)
    return 64;
  else
    return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for dgetrf based on m; 
      the return value should be multiple of 64
*/
extern "C"
int magma_get_dgetrf_nb(int m){
  if (m <= 2048)
    return 64;
  else
    return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sgeqlf based on m
*/
extern "C"
int magma_get_dgeqlf_nb(int m){
  if (m <= 1024)
    return 32;
  else if (m<=4032)
    return 64;
  else
    return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for sgeqlf based on m
*/
extern "C"
int magma_get_dgelqf_nb(int m){
  if (m <= 2048)
    return 32;
  else if (m<=4032)
    return 64;
  else
    return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for dgehrd based on m;
      the return value should be a multiple of 32
*/
extern "C"
int magma_get_dgehrd_nb(int m){
  if (m <= 2048)
    return 32;
  else
    return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for dgehrd based on m;
      the return value should be a multiple of 32
*/
extern "C"
int magma_get_sgehrd_nb(int m){
  if (m <= 1024)
    return 32;
  else
    return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for cgeqrf based on m
*/
extern "C"
int magma_get_cgeqrf_nb(int m){
  if (m <= 2048)
    return 32;
  else if (m<=4032)
    return 64;
  else
    return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for cgetrf based on m;
      the return value should be multiple of 64
*/
extern "C"
int magma_get_cgetrf_nb(int m){
  if (m <= 2048)
    return 64;
  else
    return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for cpotrf based on n
*/
extern "C"
int magma_get_cpotrf_nb(int n){
  return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for zpotrf based on n
*/
extern "C"
int magma_get_zpotrf_nb(int n){
  return 64;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for spotrf based on n
*/
extern "C"
int magma_get_zgetrf_nb(int n){
  return 128;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Return nb for cgeqrf based on m
*/
extern "C"
int magma_get_zgeqrf_nb(int m){
  if (m <= 1024)
    return 64;
  else
    return 128;
}
