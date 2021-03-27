/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) rockit_model_impl_dae_fun_jac_x_xdot_u_z_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_sq CASADI_PREFIX(sq)
#define casadi_trans CASADI_PREFIX(trans)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_sq(casadi_real x) { return x*x;}

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

void casadi_trans(const casadi_real* x, const casadi_int* sp_x, casadi_real* y,
    const casadi_int* sp_y, casadi_int* tmp) {
  casadi_int ncol_x, nnz_x, ncol_y, k;
  const casadi_int* row_x, *colind_y;
  ncol_x = sp_x[1];
  nnz_x = sp_x[2 + ncol_x];
  row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2;
  for (k=0; k<ncol_y; ++k) tmp[k] = colind_y[k];
  for (k=0; k<nnz_x; ++k) {
    y[tmp[row_x[k]]++] = x[k];
  }
}

static const casadi_int casadi_s0[12] = {6, 6, 0, 1, 1, 3, 3, 3, 3, 5, 0, 1};
static const casadi_int casadi_s1[12] = {6, 6, 0, 1, 2, 2, 2, 2, 3, 2, 2, 0};
static const casadi_int casadi_s2[15] = {6, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s3[12] = {6, 4, 0, 2, 3, 4, 5, 0, 1, 2, 3, 4};
static const casadi_int casadi_s4[14] = {4, 6, 0, 1, 2, 3, 4, 5, 5, 0, 0, 1, 2, 3};
static const casadi_int casadi_s5[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s6[8] = {4, 1, 0, 4, 0, 1, 2, 3};
static const casadi_int casadi_s7[3] = {0, 0, 0};
static const casadi_int casadi_s8[3] = {6, 0, 0};

/* rockit_model_impl_dae_fun_jac_x_xdot_u_z:(i0[6],i1[6],i2[4],i3[],i4[])->(o0[6],o1[6x6,3nz],o2[6x6,6nz],o3[6x4,5nz],o4[6x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real *rr, *ss;
  const casadi_real *cs;
  casadi_real *w0=w+0, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, *w12=w+17, *w13=w+23, *w17=w+26, *w18=w+29, *w19=w+32, *w20=w+37, *w21=w+41, *w22=w+46;
  /* #0: @0 = input[1][0] */
  casadi_copy(arg[1], 6, w0);
  /* #1: @1 = input[2][0] */
  w1 = arg[2] ? arg[2][0] : 0;
  /* #2: @2 = input[0][2] */
  w2 = arg[0] ? arg[0][2] : 0;
  /* #3: @3 = cos(@2) */
  w3 = cos( w2 );
  /* #4: @4 = (@1*@3) */
  w4  = (w1*w3);
  /* #5: @5 = sin(@2) */
  w5 = sin( w2 );
  /* #6: @6 = (@1*@5) */
  w6  = (w1*w5);
  /* #7: @7 = input[2][1] */
  w7 = arg[2] ? arg[2][1] : 0;
  /* #8: @8 = input[2][2] */
  w8 = arg[2] ? arg[2][2] : 0;
  /* #9: @9 = input[2][3] */
  w9 = arg[2] ? arg[2][3] : 0;
  /* #10: @10 = input[0][0] */
  w10 = arg[0] ? arg[0][0] : 0;
  /* #11: @11 = sq(@10) */
  w11 = casadi_sq( w10 );
  /* #12: @12 = vertcat(@4, @6, @7, @8, @9, @11) */
  rr=w12;
  *rr++ = w4;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  *rr++ = w9;
  *rr++ = w11;
  /* #13: @0 = (@0-@12) */
  for (i=0, rr=w0, cs=w12; i<6; ++i) (*rr++) -= (*cs++);
  /* #14: output[0][0] = @0 */
  casadi_copy(w0, 6, res[0]);
  /* #15: @13 = zeros(6x6,3nz) */
  casadi_clear(w13, 3);
  /* #16: @4 = sin(@2) */
  w4 = sin( w2 );
  /* #17: @0 = ones(6x1) */
  casadi_fill(w0, 6, 1.);
  /* #18: {@6, NULL, @7, NULL, NULL, NULL} = vertsplit(@0) */
  w6 = w0[0];
  w7 = w0[2];
  /* #19: @4 = (@4*@7) */
  w4 *= w7;
  /* #20: @4 = (@1*@4) */
  w4  = (w1*w4);
  /* #21: @4 = (-@4) */
  w4 = (- w4 );
  /* #22: @2 = cos(@2) */
  w2 = cos( w2 );
  /* #23: @2 = (@2*@7) */
  w2 *= w7;
  /* #24: @1 = (@1*@2) */
  w1 *= w2;
  /* #25: @14 = 00 */
  /* #26: @15 = 00 */
  /* #27: @16 = 00 */
  /* #28: @10 = (2.*@10) */
  w10 = (2.* w10 );
  /* #29: @10 = (@10*@6) */
  w10 *= w6;
  /* #30: @17 = vertcat(@4, @1, @14, @15, @16, @10) */
  rr=w17;
  *rr++ = w4;
  *rr++ = w1;
  *rr++ = w10;
  /* #31: @17 = (-@17) */
  for (i=0, rr=w17, cs=w17; i<3; ++i) *rr++ = (- *cs++ );
  /* #32: @18 = @17[:3] */
  for (rr=w18, ss=w17+0; ss!=w17+3; ss+=1) *rr++ = *ss;
  /* #33: (@13[:3] = @18) */
  for (rr=w13+0, ss=w18; rr!=w13+3; rr+=1) *rr = *ss++;
  /* #34: @18 = @13' */
  casadi_trans(w13,casadi_s1, w18, casadi_s0, iw);
  /* #35: output[1][0] = @18 */
  casadi_copy(w18, 3, res[1]);
  /* #36: @0 = zeros(6x6,6nz) */
  casadi_clear(w0, 6);
  /* #37: @12 = ones(6x1) */
  casadi_fill(w12, 6, 1.);
  /* #38: (@0[:6] = @12) */
  for (rr=w0+0, ss=w12; rr!=w0+6; rr+=1) *rr = *ss++;
  /* #39: @12 = @0' */
  casadi_trans(w0,casadi_s2, w12, casadi_s2, iw);
  /* #40: output[2][0] = @12 */
  casadi_copy(w12, 6, res[2]);
  /* #41: @19 = zeros(4x6,5nz) */
  casadi_clear(w19, 5);
  /* #42: @20 = ones(4x1) */
  casadi_fill(w20, 4, 1.);
  /* #43: {@4, @1, @10, @6} = vertsplit(@20) */
  w4 = w20[0];
  w1 = w20[1];
  w10 = w20[2];
  w6 = w20[3];
  /* #44: @3 = (@3*@4) */
  w3 *= w4;
  /* #45: @5 = (@5*@4) */
  w5 *= w4;
  /* #46: @14 = 00 */
  /* #47: @21 = vertcat(@3, @5, @1, @10, @6, @14) */
  rr=w21;
  *rr++ = w3;
  *rr++ = w5;
  *rr++ = w1;
  *rr++ = w10;
  *rr++ = w6;
  /* #48: @21 = (-@21) */
  for (i=0, rr=w21, cs=w21; i<5; ++i) *rr++ = (- *cs++ );
  /* #49: @22 = @21[:5] */
  for (rr=w22, ss=w21+0; ss!=w21+5; ss+=1) *rr++ = *ss;
  /* #50: (@19[:5] = @22) */
  for (rr=w19+0, ss=w22; rr!=w19+5; rr+=1) *rr = *ss++;
  /* #51: @22 = @19' */
  casadi_trans(w19,casadi_s4, w22, casadi_s3, iw);
  /* #52: output[3][0] = @22 */
  casadi_copy(w22, 5, res[3]);
  return 0;
}

CASADI_SYMBOL_EXPORT int rockit_model_impl_dae_fun_jac_x_xdot_u_z(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int rockit_model_impl_dae_fun_jac_x_xdot_u_z_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int rockit_model_impl_dae_fun_jac_x_xdot_u_z_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void rockit_model_impl_dae_fun_jac_x_xdot_u_z_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int rockit_model_impl_dae_fun_jac_x_xdot_u_z_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void rockit_model_impl_dae_fun_jac_x_xdot_u_z_release(int mem) {
}

CASADI_SYMBOL_EXPORT void rockit_model_impl_dae_fun_jac_x_xdot_u_z_incref(void) {
}

CASADI_SYMBOL_EXPORT void rockit_model_impl_dae_fun_jac_x_xdot_u_z_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int rockit_model_impl_dae_fun_jac_x_xdot_u_z_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int rockit_model_impl_dae_fun_jac_x_xdot_u_z_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real rockit_model_impl_dae_fun_jac_x_xdot_u_z_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* rockit_model_impl_dae_fun_jac_x_xdot_u_z_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* rockit_model_impl_dae_fun_jac_x_xdot_u_z_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* rockit_model_impl_dae_fun_jac_x_xdot_u_z_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s5;
    case 1: return casadi_s5;
    case 2: return casadi_s6;
    case 3: return casadi_s7;
    case 4: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* rockit_model_impl_dae_fun_jac_x_xdot_u_z_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s5;
    case 1: return casadi_s0;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    case 4: return casadi_s8;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int rockit_model_impl_dae_fun_jac_x_xdot_u_z_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 11;
  if (sz_res) *sz_res = 11;
  if (sz_iw) *sz_iw = 7;
  if (sz_w) *sz_w = 51;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif