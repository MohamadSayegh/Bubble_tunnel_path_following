/*
 * Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
 * Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
 * Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
 * Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */

// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "rockit_model_model/rockit_model_model.h"



#include "rockit_model_constraints/rockit_model_h_constraint.h"


#include "rockit_model_constraints/rockit_model_h_e_constraint.h"

#include "rockit_model_cost/rockit_model_external_cost.h"

#include "rockit_model_cost/rockit_model_external_cost_e.h"


#include "acados_solver_rockit_model.h"

#define NX     6
#define NZ     0
#define NU     4
#define NP     12
#define NBX    0
#define NBX0   6
#define NBU    4
#define NSBX   0
#define NSBU   0
#define NSH    0
#define NSG    0
#define NSPHI  0
#define NSHN   0
#define NSGN   4
#define NSPHIN 0
#define NSBXN  0
#define NS     0
#define NSN    4
#define NG     0
#define NBXN   0
#define NGN    4
#define NY     0
#define NYN    0
#define N      5
#define NH     1
#define NPHI   0
#define NHN    1
#define NPHIN  0
#define NR     0


// ** solver data **

nlp_solver_capsule * rockit_model_acados_create_capsule()
{
    void* capsule_mem = malloc(sizeof(nlp_solver_capsule));
    nlp_solver_capsule *capsule = (nlp_solver_capsule *) capsule_mem;

    return capsule;
}


int rockit_model_acados_free_capsule(nlp_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int rockit_model_acados_create(nlp_solver_capsule * capsule)
{
    int status = 0;

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    /************************************************
    *  plan & config
    ************************************************/
    ocp_nlp_plan * nlp_solver_plan = ocp_nlp_plan_create(N);
    capsule->nlp_solver_plan = nlp_solver_plan;
    nlp_solver_plan->nlp_solver = SQP;
    

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = FULL_CONDENSING_HPIPM;
    for (int i = 0; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = EXTERNAL;

    nlp_solver_plan->nlp_cost[N] = EXTERNAL;

    for (int i = 0; i < N; i++)
    {
        
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = ERK;
    }

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;
    nlp_solver_plan->regularization = MIRROR;
    ocp_nlp_config * nlp_config = ocp_nlp_config_create(*nlp_solver_plan);
    capsule->nlp_config = nlp_config;


    /************************************************
    *  dimensions
    ************************************************/
    int nx[N+1];
    int nu[N+1];
    int nbx[N+1];
    int nbu[N+1];
    int nsbx[N+1];
    int nsbu[N+1];
    int nsg[N+1];
    int nsh[N+1];
    int nsphi[N+1];
    int ns[N+1];
    int ng[N+1];
    int nh[N+1];
    int nphi[N+1];
    int nz[N+1];
    int ny[N+1];
    int nr[N+1];
    int nbxe[N+1];

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i] = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
    }

    // for initial state
    nbx[0]  = NBX0;
    nsbx[0] = 0;
    ns[0] = NS - NSBX;
    nbxe[0] = 0;

    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);
    capsule->nlp_dims = nlp_dims;

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);



    /************************************************
    *  external functions
    ************************************************/
    capsule->nl_constr_h_fun_jac = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->nl_constr_h_fun_jac[i].casadi_fun = &rockit_model_constr_h_fun_jac_uxt_zt;
        capsule->nl_constr_h_fun_jac[i].casadi_n_in = &rockit_model_constr_h_fun_jac_uxt_zt_n_in;
        capsule->nl_constr_h_fun_jac[i].casadi_n_out = &rockit_model_constr_h_fun_jac_uxt_zt_n_out;
        capsule->nl_constr_h_fun_jac[i].casadi_sparsity_in = &rockit_model_constr_h_fun_jac_uxt_zt_sparsity_in;
        capsule->nl_constr_h_fun_jac[i].casadi_sparsity_out = &rockit_model_constr_h_fun_jac_uxt_zt_sparsity_out;
        capsule->nl_constr_h_fun_jac[i].casadi_work = &rockit_model_constr_h_fun_jac_uxt_zt_work;
        external_function_param_casadi_create(&capsule->nl_constr_h_fun_jac[i], 12);
    }
    capsule->nl_constr_h_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->nl_constr_h_fun[i].casadi_fun = &rockit_model_constr_h_fun;
        capsule->nl_constr_h_fun[i].casadi_n_in = &rockit_model_constr_h_fun_n_in;
        capsule->nl_constr_h_fun[i].casadi_n_out = &rockit_model_constr_h_fun_n_out;
        capsule->nl_constr_h_fun[i].casadi_sparsity_in = &rockit_model_constr_h_fun_sparsity_in;
        capsule->nl_constr_h_fun[i].casadi_sparsity_out = &rockit_model_constr_h_fun_sparsity_out;
        capsule->nl_constr_h_fun[i].casadi_work = &rockit_model_constr_h_fun_work;
        external_function_param_casadi_create(&capsule->nl_constr_h_fun[i], 12);
    }
    
    capsule->nl_constr_h_fun_jac_hess = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->nl_constr_h_fun_jac_hess[i].casadi_fun = &rockit_model_constr_h_fun_jac_uxt_hess;
        capsule->nl_constr_h_fun_jac_hess[i].casadi_n_in = &rockit_model_constr_h_fun_jac_uxt_hess_n_in;
        capsule->nl_constr_h_fun_jac_hess[i].casadi_n_out = &rockit_model_constr_h_fun_jac_uxt_hess_n_out;
        capsule->nl_constr_h_fun_jac_hess[i].casadi_sparsity_in = &rockit_model_constr_h_fun_jac_uxt_hess_sparsity_in;
        capsule->nl_constr_h_fun_jac_hess[i].casadi_sparsity_out = &rockit_model_constr_h_fun_jac_uxt_hess_sparsity_out;
        capsule->nl_constr_h_fun_jac_hess[i].casadi_work = &rockit_model_constr_h_fun_jac_uxt_hess_work;

        external_function_param_casadi_create(&capsule->nl_constr_h_fun_jac_hess[i], 12);
    }
    
    
    capsule->nl_constr_h_e_fun_jac.casadi_fun = &rockit_model_constr_h_e_fun_jac_uxt_zt;
    capsule->nl_constr_h_e_fun_jac.casadi_n_in = &rockit_model_constr_h_e_fun_jac_uxt_zt_n_in;
    capsule->nl_constr_h_e_fun_jac.casadi_n_out = &rockit_model_constr_h_e_fun_jac_uxt_zt_n_out;
    capsule->nl_constr_h_e_fun_jac.casadi_sparsity_in = &rockit_model_constr_h_e_fun_jac_uxt_zt_sparsity_in;
    capsule->nl_constr_h_e_fun_jac.casadi_sparsity_out = &rockit_model_constr_h_e_fun_jac_uxt_zt_sparsity_out;
    capsule->nl_constr_h_e_fun_jac.casadi_work = &rockit_model_constr_h_e_fun_jac_uxt_zt_work;
    external_function_param_casadi_create(&capsule->nl_constr_h_e_fun_jac, 12);

    capsule->nl_constr_h_e_fun.casadi_fun = &rockit_model_constr_h_e_fun;
    capsule->nl_constr_h_e_fun.casadi_n_in = &rockit_model_constr_h_e_fun_n_in;
    capsule->nl_constr_h_e_fun.casadi_n_out = &rockit_model_constr_h_e_fun_n_out;
    capsule->nl_constr_h_e_fun.casadi_sparsity_in = &rockit_model_constr_h_e_fun_sparsity_in;
    capsule->nl_constr_h_e_fun.casadi_sparsity_out = &rockit_model_constr_h_e_fun_sparsity_out;
    capsule->nl_constr_h_e_fun.casadi_work = &rockit_model_constr_h_e_fun_work;
    external_function_param_casadi_create(&capsule->nl_constr_h_e_fun, 12);

    
    capsule->nl_constr_h_e_fun_jac_hess.casadi_fun = &rockit_model_constr_h_e_fun_jac_uxt_hess;
    capsule->nl_constr_h_e_fun_jac_hess.casadi_n_in = &rockit_model_constr_h_e_fun_jac_uxt_hess_n_in;
    capsule->nl_constr_h_e_fun_jac_hess.casadi_n_out = &rockit_model_constr_h_e_fun_jac_uxt_hess_n_out;
    capsule->nl_constr_h_e_fun_jac_hess.casadi_sparsity_in = &rockit_model_constr_h_e_fun_jac_uxt_hess_sparsity_in;
    capsule->nl_constr_h_e_fun_jac_hess.casadi_sparsity_out = &rockit_model_constr_h_e_fun_jac_uxt_hess_sparsity_out;
    capsule->nl_constr_h_e_fun_jac_hess.casadi_work = &rockit_model_constr_h_e_fun_jac_uxt_hess_work;
    external_function_param_casadi_create(&capsule->nl_constr_h_e_fun_jac_hess, 12);
    


    // explicit ode
    capsule->forw_vde_casadi = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->forw_vde_casadi[i].casadi_fun = &rockit_model_expl_vde_forw;
        capsule->forw_vde_casadi[i].casadi_n_in = &rockit_model_expl_vde_forw_n_in;
        capsule->forw_vde_casadi[i].casadi_n_out = &rockit_model_expl_vde_forw_n_out;
        capsule->forw_vde_casadi[i].casadi_sparsity_in = &rockit_model_expl_vde_forw_sparsity_in;
        capsule->forw_vde_casadi[i].casadi_sparsity_out = &rockit_model_expl_vde_forw_sparsity_out;
        capsule->forw_vde_casadi[i].casadi_work = &rockit_model_expl_vde_forw_work;
        external_function_param_casadi_create(&capsule->forw_vde_casadi[i], 12);
    }

    capsule->expl_ode_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->expl_ode_fun[i].casadi_fun = &rockit_model_expl_ode_fun;
        capsule->expl_ode_fun[i].casadi_n_in = &rockit_model_expl_ode_fun_n_in;
        capsule->expl_ode_fun[i].casadi_n_out = &rockit_model_expl_ode_fun_n_out;
        capsule->expl_ode_fun[i].casadi_sparsity_in = &rockit_model_expl_ode_fun_sparsity_in;
        capsule->expl_ode_fun[i].casadi_sparsity_out = &rockit_model_expl_ode_fun_sparsity_out;
        capsule->expl_ode_fun[i].casadi_work = &rockit_model_expl_ode_fun_work;
        external_function_param_casadi_create(&capsule->expl_ode_fun[i], 12);
    }
    capsule->hess_vde_casadi = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        capsule->hess_vde_casadi[i].casadi_fun = &rockit_model_expl_ode_hess;
        capsule->hess_vde_casadi[i].casadi_n_in = &rockit_model_expl_ode_hess_n_in;
        capsule->hess_vde_casadi[i].casadi_n_out = &rockit_model_expl_ode_hess_n_out;
        capsule->hess_vde_casadi[i].casadi_sparsity_in = &rockit_model_expl_ode_hess_sparsity_in;
        capsule->hess_vde_casadi[i].casadi_sparsity_out = &rockit_model_expl_ode_hess_sparsity_out;
        capsule->hess_vde_casadi[i].casadi_work = &rockit_model_expl_ode_hess_work;
        external_function_param_casadi_create(&capsule->hess_vde_casadi[i], 12);
    }


    // external cost
    capsule->ext_cost_fun = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        capsule->ext_cost_fun[i].casadi_fun = &rockit_model_cost_ext_cost_fun;
        capsule->ext_cost_fun[i].casadi_n_in = &rockit_model_cost_ext_cost_fun_n_in;
        capsule->ext_cost_fun[i].casadi_n_out = &rockit_model_cost_ext_cost_fun_n_out;
        capsule->ext_cost_fun[i].casadi_sparsity_in = &rockit_model_cost_ext_cost_fun_sparsity_in;
        capsule->ext_cost_fun[i].casadi_sparsity_out = &rockit_model_cost_ext_cost_fun_sparsity_out;
        capsule->ext_cost_fun[i].casadi_work = &rockit_model_cost_ext_cost_fun_work;

        external_function_param_casadi_create(&capsule->ext_cost_fun[i], 12);
    }

    capsule->ext_cost_fun_jac = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        // residual function
        capsule->ext_cost_fun_jac[i].casadi_fun = &rockit_model_cost_ext_cost_fun_jac;
        capsule->ext_cost_fun_jac[i].casadi_n_in = &rockit_model_cost_ext_cost_fun_jac_n_in;
        capsule->ext_cost_fun_jac[i].casadi_n_out = &rockit_model_cost_ext_cost_fun_jac_n_out;
        capsule->ext_cost_fun_jac[i].casadi_sparsity_in = &rockit_model_cost_ext_cost_fun_jac_sparsity_in;
        capsule->ext_cost_fun_jac[i].casadi_sparsity_out = &rockit_model_cost_ext_cost_fun_jac_sparsity_out;
        capsule->ext_cost_fun_jac[i].casadi_work = &rockit_model_cost_ext_cost_fun_jac_work;

        external_function_param_casadi_create(&capsule->ext_cost_fun_jac[i], 12);
    }

    capsule->ext_cost_fun_jac_hess = (external_function_param_casadi *) malloc(sizeof(external_function_param_casadi)*N);
    for (int i = 0; i < N; i++)
    {
        // residual function
        capsule->ext_cost_fun_jac_hess[i].casadi_fun = &rockit_model_cost_ext_cost_fun_jac_hess;
        capsule->ext_cost_fun_jac_hess[i].casadi_n_in = &rockit_model_cost_ext_cost_fun_jac_hess_n_in;
        capsule->ext_cost_fun_jac_hess[i].casadi_n_out = &rockit_model_cost_ext_cost_fun_jac_hess_n_out;
        capsule->ext_cost_fun_jac_hess[i].casadi_sparsity_in = &rockit_model_cost_ext_cost_fun_jac_hess_sparsity_in;
        capsule->ext_cost_fun_jac_hess[i].casadi_sparsity_out = &rockit_model_cost_ext_cost_fun_jac_hess_sparsity_out;
        capsule->ext_cost_fun_jac_hess[i].casadi_work = &rockit_model_cost_ext_cost_fun_jac_hess_work;

        external_function_param_casadi_create(&capsule->ext_cost_fun_jac_hess[i], 12);
    }
    // external cost
    capsule->ext_cost_e_fun.casadi_fun = &rockit_model_cost_ext_cost_e_fun;
    capsule->ext_cost_e_fun.casadi_n_in = &rockit_model_cost_ext_cost_e_fun_n_in;
    capsule->ext_cost_e_fun.casadi_n_out = &rockit_model_cost_ext_cost_e_fun_n_out;
    capsule->ext_cost_e_fun.casadi_sparsity_in = &rockit_model_cost_ext_cost_e_fun_sparsity_in;
    capsule->ext_cost_e_fun.casadi_sparsity_out = &rockit_model_cost_ext_cost_e_fun_sparsity_out;
    capsule->ext_cost_e_fun.casadi_work = &rockit_model_cost_ext_cost_e_fun_work;
    external_function_param_casadi_create(&capsule->ext_cost_e_fun, 12);

    // external cost
    capsule->ext_cost_e_fun_jac.casadi_fun = &rockit_model_cost_ext_cost_e_fun_jac;
    capsule->ext_cost_e_fun_jac.casadi_n_in = &rockit_model_cost_ext_cost_e_fun_jac_n_in;
    capsule->ext_cost_e_fun_jac.casadi_n_out = &rockit_model_cost_ext_cost_e_fun_jac_n_out;
    capsule->ext_cost_e_fun_jac.casadi_sparsity_in = &rockit_model_cost_ext_cost_e_fun_jac_sparsity_in;
    capsule->ext_cost_e_fun_jac.casadi_sparsity_out = &rockit_model_cost_ext_cost_e_fun_jac_sparsity_out;
    capsule->ext_cost_e_fun_jac.casadi_work = &rockit_model_cost_ext_cost_e_fun_jac_work;
    external_function_param_casadi_create(&capsule->ext_cost_e_fun_jac, 12);

    // external cost
    capsule->ext_cost_e_fun_jac_hess.casadi_fun = &rockit_model_cost_ext_cost_e_fun_jac_hess;
    capsule->ext_cost_e_fun_jac_hess.casadi_n_in = &rockit_model_cost_ext_cost_e_fun_jac_hess_n_in;
    capsule->ext_cost_e_fun_jac_hess.casadi_n_out = &rockit_model_cost_ext_cost_e_fun_jac_hess_n_out;
    capsule->ext_cost_e_fun_jac_hess.casadi_sparsity_in = &rockit_model_cost_ext_cost_e_fun_jac_hess_sparsity_in;
    capsule->ext_cost_e_fun_jac_hess.casadi_sparsity_out = &rockit_model_cost_ext_cost_e_fun_jac_hess_sparsity_out;
    capsule->ext_cost_e_fun_jac_hess.casadi_work = &rockit_model_cost_ext_cost_e_fun_jac_hess_work;
    external_function_param_casadi_create(&capsule->ext_cost_e_fun_jac_hess, 12);

    /************************************************
    *  nlp_in
    ************************************************/
    ocp_nlp_in * nlp_in = ocp_nlp_in_create(nlp_config, nlp_dims);
    capsule->nlp_in = nlp_in;

    double time_steps[N];
    time_steps[0] = 2;
    time_steps[1] = 2;
    time_steps[2] = 2.000000000000001;
    time_steps[3] = 1.9999999999999991;
    time_steps[4] = 2;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &time_steps[i]);
    }

    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "expl_vde_forw", &capsule->forw_vde_casadi[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "expl_ode_fun", &capsule->expl_ode_fun[i]);
        ocp_nlp_dynamics_model_set(nlp_config, nlp_dims, nlp_in, i, "expl_ode_hess", &capsule->hess_vde_casadi[i]);
    
    }


    /**** Cost ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun", &capsule->ext_cost_fun[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac", &capsule->ext_cost_fun_jac[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "ext_cost_fun_jac_hess", &capsule->ext_cost_fun_jac_hess[i]);
    }




    // terminal cost

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun", &capsule->ext_cost_e_fun);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun_jac", &capsule->ext_cost_e_fun_jac);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "ext_cost_fun_jac_hess", &capsule->ext_cost_e_fun_jac_hess);


    double Zl_e[NSN];
    double Zu_e[NSN];
    double zl_e[NSN];
    double zu_e[NSN];

    
    Zl_e[0] = 0;
    Zl_e[1] = 0;
    Zl_e[2] = 0;
    Zl_e[3] = 0;

    
    Zu_e[0] = 0;
    Zu_e[1] = 0;
    Zu_e[2] = 0;
    Zu_e[3] = 0;

    
    zl_e[0] = 0;
    zl_e[1] = 0;
    zl_e[2] = 0;
    zl_e[3] = 0;

    
    zu_e[0] = 1;
    zu_e[1] = 1;
    zu_e[2] = 1;
    zu_e[3] = 1;

    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zl", Zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "Zu", Zu_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zl", zl_e);
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "zu", zu_e);

    /**** Constraints ****/

    // bounds for initial stage

    // x0
    int idxbx0[6];
    
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;

    double lbx0[6];
    double ubx0[6];
    
    lbx0[0] = 0;
    ubx0[0] = 0;
    lbx0[1] = 0;
    ubx0[1] = 0;
    lbx0[2] = 0;
    ubx0[2] = 0;
    lbx0[3] = 0;
    ubx0[3] = 0;
    lbx0[4] = 0;
    ubx0[4] = 0;
    lbx0[5] = 0;
    ubx0[5] = 0;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);



    /* constraints that are the same for initial and intermediate */



    // u
    int idxbu[NBU];
    
    idxbu[0] = 0;
    idxbu[1] = 1;
    idxbu[2] = 2;
    idxbu[3] = 3;
    double lbu[NBU];
    double ubu[NBU];
    
    lbu[0] = 0;
    ubu[0] = 1;
    lbu[1] = -3.141592653589793;
    ubu[1] = 3.141592653589793;
    lbu[2] = 0;
    ubu[2] = 100000;
    lbu[3] = 0;
    ubu[3] = 100000;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubu", ubu);
    }















    // set up nonlinear constraints for stage 0 to N-1 
    double lh[NH];
    double uh[NH];

    
    lh[0] = -100000;

    
    uh[0] = 0;
    
    for (int i = 0; i < N; i++)
    {
        // nonlinear constraints for stages 0 to N-1
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun_jac",
                                      &capsule->nl_constr_h_fun_jac[i]);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun",
                                      &capsule->nl_constr_h_fun[i]);
        
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i,
                                      "nl_constr_h_fun_jac_hess", &capsule->nl_constr_h_fun_jac_hess[i]);
        
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "uh", uh);
    }




    /* terminal constraints */



    // set up soft bounds for general linear constraints
    int idxsg_e[NSGN];
    
    idxsg_e[0] = 1;
    idxsg_e[1] = 3;
    idxsg_e[2] = 0;
    idxsg_e[3] = 2;
    double lsg_e[NSGN];
    double usg_e[NSGN];
    
    lsg_e[0] = 0;
    usg_e[0] = 0;
    lsg_e[1] = 0;
    usg_e[1] = 0;
    lsg_e[2] = 0;
    usg_e[2] = 0;
    lsg_e[3] = 0;
    usg_e[3] = 0;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxsg", idxsg_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lsg", lsg_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "usg", usg_e);








    // set up general constraints for last stage 
    double C_e[NGN*NX];
    double lg_e[NGN];
    double ug_e[NGN];

    

    
    lg_e[0] = -100000;
    ug_e[0] = -0.9984141238521388;
    lg_e[1] = -100000;
    ug_e[1] = 0.9984141238521388;
    lg_e[2] = -100000;
    ug_e[2] = -4.990381719334589;
    lg_e[3] = -100000;
    ug_e[3] = 4.990381719334589;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "C", C_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lg", lg_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "ug", ug_e);


    // set up nonlinear constraints for last stage 
    double lh_e[NHN];
    double uh_e[NHN];

    
    lh_e[0] = -100000;

    
    uh_e[0] = 0;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun_jac", &capsule->nl_constr_h_e_fun_jac);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun", &capsule->nl_constr_h_e_fun);
    
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "nl_constr_h_fun_jac_hess",
                                  &capsule->nl_constr_h_e_fun_jac_hess);
    
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lh", lh_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "uh", uh_e);




    /************************************************
    *  opts
    ************************************************/

    capsule->nlp_opts = ocp_nlp_solver_opts_create(nlp_config, nlp_dims);


    bool nlp_solver_exact_hessian = true;
    // TODO: this if should not be needed! however, calling the setter with false leads to weird behavior. Investigate!
    if (nlp_solver_exact_hessian)
    {
        ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "exact_hess", &nlp_solver_exact_hessian);
    }
    int exact_hess_dyn = 1;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "exact_hess_dyn", &exact_hess_dyn);

    int exact_hess_cost = 1;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "exact_hess_cost", &exact_hess_cost);

    int exact_hess_constr = 1;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "exact_hess_constr", &exact_hess_constr);
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization", "fixed_step");
    int num_steps_val = 1;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_num_steps", &num_steps_val);

    int ns_val = 4;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_num_stages", &ns_val);

    int newton_iter_val = 3;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);

    bool tmp_bool = false;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

    double nlp_solver_step_length = 1;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "step_length", &nlp_solver_step_length);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */

    int qp_solver_iter_max = 1000;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "qp_iter_max", &qp_solver_iter_max);
    // set SQP specific options
    double nlp_solver_tol_stat = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "tol_stat", &nlp_solver_tol_stat);

    double nlp_solver_tol_eq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "tol_eq", &nlp_solver_tol_eq);

    double nlp_solver_tol_ineq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "tol_ineq", &nlp_solver_tol_ineq);

    double nlp_solver_tol_comp = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "tol_comp", &nlp_solver_tol_comp);

    int nlp_solver_max_iter = 500;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "max_iter", &nlp_solver_max_iter);

    int initialize_t_slacks = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "initialize_t_slacks", &initialize_t_slacks);

    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "print_level", &print_level);


    int ext_cost_num_hess = 0;
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, i, "cost_numerical_hessian", &ext_cost_num_hess);
    }
    ocp_nlp_solver_opts_set_at_stage(nlp_config, capsule->nlp_opts, N, "cost_numerical_hessian", &ext_cost_num_hess);


    /* out */
    ocp_nlp_out * nlp_out = ocp_nlp_out_create(nlp_config, nlp_dims);
    capsule->nlp_out = nlp_out;

    // initialize primal solution
    double x0[6];

    // initialize with x0
    
    x0[0] = 0;
    x0[1] = 0;
    x0[2] = 0;
    x0[3] = 0;
    x0[4] = 0;
    x0[5] = 0;


    double u0[NU];
    
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;
    u0[3] = 0.0;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x0);
    
    capsule->nlp_solver = ocp_nlp_solver_create(nlp_config, nlp_dims, capsule->nlp_opts);



    // initialize parameters to nominal value
    double p[12];
    
    p[0] = 0;
    p[1] = 0;
    p[2] = 0;
    p[3] = 0;
    p[4] = 0;
    p[5] = 0.9984141238521388;
    p[6] = 4.990381719334589;
    p[7] = 0.012865463007186105;
    p[8] = 0.031157516773728397;
    p[9] = 2.691048944219061;
    p[10] = 0.012865463007186105;
    p[11] = 0.031157516773728407;

    for (int i = 0; i <= N; i++)
    {
        rockit_model_acados_update_params(capsule, i, p, NP);
    }

    status = ocp_nlp_precompute(capsule->nlp_solver, nlp_in, nlp_out);

    if (status != ACADOS_SUCCESS)
    {
        printf("\nocp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int rockit_model_acados_update_params(nlp_solver_capsule * capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 12;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    if (stage < 5)
    {
        capsule->forw_vde_casadi[stage].set_param(capsule->forw_vde_casadi+stage, p);
        capsule->expl_ode_fun[stage].set_param(capsule->expl_ode_fun+stage, p);
        capsule->hess_vde_casadi[stage].set_param(capsule->hess_vde_casadi+stage, p);
    

        // constraints
    
        capsule->nl_constr_h_fun_jac[stage].set_param(capsule->nl_constr_h_fun_jac+stage, p);
        capsule->nl_constr_h_fun[stage].set_param(capsule->nl_constr_h_fun+stage, p);
        capsule->nl_constr_h_fun_jac_hess[stage].set_param(capsule->nl_constr_h_fun_jac_hess+stage, p);

        // cost
        capsule->ext_cost_fun[stage].set_param(capsule->ext_cost_fun+stage, p);
        capsule->ext_cost_fun_jac[stage].set_param(capsule->ext_cost_fun_jac+stage, p);
        capsule->ext_cost_fun_jac_hess[stage].set_param(capsule->ext_cost_fun_jac_hess+stage, p);

    }
    else // stage == N
    {
        // terminal shooting node has no dynamics
        // cost
        capsule->ext_cost_e_fun.set_param(&capsule->ext_cost_e_fun, p);
        capsule->ext_cost_e_fun_jac.set_param(&capsule->ext_cost_e_fun_jac, p);
    //
        capsule->ext_cost_e_fun_jac_hess.set_param(&capsule->ext_cost_e_fun_jac_hess, p);
    //
    
        // constraints
    
        capsule->nl_constr_h_e_fun_jac.set_param(&capsule->nl_constr_h_e_fun_jac, p);
        capsule->nl_constr_h_e_fun.set_param(&capsule->nl_constr_h_e_fun, p);
        capsule->nl_constr_h_e_fun_jac_hess.set_param(&capsule->nl_constr_h_e_fun_jac_hess, p);
    
    }


    return solver_status;
}



int rockit_model_acados_solve(nlp_solver_capsule * capsule)
{
    // solve NLP 
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}


int rockit_model_acados_free(nlp_solver_capsule * capsule)
{
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < 5; i++)
    {
        external_function_param_casadi_free(&capsule->forw_vde_casadi[i]);
        external_function_param_casadi_free(&capsule->expl_ode_fun[i]);
        external_function_param_casadi_free(&capsule->hess_vde_casadi[i]);
    }
    free(capsule->forw_vde_casadi);
    free(capsule->expl_ode_fun);
    free(capsule->hess_vde_casadi);

    // cost
    for (int i = 0; i < 5; i++)
    {
        external_function_param_casadi_free(&capsule->ext_cost_fun[i]);
        external_function_param_casadi_free(&capsule->ext_cost_fun_jac[i]);
        external_function_param_casadi_free(&capsule->ext_cost_fun_jac_hess[i]);
    }
    free(capsule->ext_cost_fun);
    free(capsule->ext_cost_fun_jac);
    free(capsule->ext_cost_fun_jac_hess);
    external_function_param_casadi_free(&capsule->ext_cost_e_fun);
    external_function_param_casadi_free(&capsule->ext_cost_e_fun_jac);
    external_function_param_casadi_free(&capsule->ext_cost_e_fun_jac_hess);

    // constraints
    for (int i = 0; i < 5; i++)
    {
        external_function_param_casadi_free(&capsule->nl_constr_h_fun_jac[i]);
        external_function_param_casadi_free(&capsule->nl_constr_h_fun[i]);
    }
    for (int i = 0; i < 5; i++)
    {
        external_function_param_casadi_free(&capsule->nl_constr_h_fun_jac_hess[i]);
    }
    free(capsule->nl_constr_h_fun_jac);
    free(capsule->nl_constr_h_fun);
    free(capsule->nl_constr_h_fun_jac_hess);
    external_function_param_casadi_free(&capsule->nl_constr_h_e_fun_jac);
    external_function_param_casadi_free(&capsule->nl_constr_h_e_fun);
    external_function_param_casadi_free(&capsule->nl_constr_h_e_fun_jac_hess);

    return 0;
}

ocp_nlp_in *rockit_model_acados_get_nlp_in(nlp_solver_capsule * capsule) { return capsule->nlp_in; }
ocp_nlp_out *rockit_model_acados_get_nlp_out(nlp_solver_capsule * capsule) { return capsule->nlp_out; }
ocp_nlp_solver *rockit_model_acados_get_nlp_solver(nlp_solver_capsule * capsule) { return capsule->nlp_solver; }
ocp_nlp_config *rockit_model_acados_get_nlp_config(nlp_solver_capsule * capsule) { return capsule->nlp_config; }
void *rockit_model_acados_get_nlp_opts(nlp_solver_capsule * capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *rockit_model_acados_get_nlp_dims(nlp_solver_capsule * capsule) { return capsule->nlp_dims; }
ocp_nlp_plan *rockit_model_acados_get_nlp_plan(nlp_solver_capsule * capsule) { return capsule->nlp_solver_plan; }


void rockit_model_acados_print_stats(nlp_solver_capsule * capsule)
{
    int sqp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "sqp_iter", &sqp_iter);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "stat_m", &stat_m);

    
    double stat[5000];
    ocp_nlp_get(capsule->nlp_config, capsule->nlp_solver, "statistics", stat);

    int nrow = sqp_iter+1 < stat_m ? sqp_iter+1 : stat_m;

    printf("iter\tres_stat\tres_eq\t\tres_ineq\tres_comp\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            if (j == 0 || j > 4)
            {
                tmp_int = (int) stat[i + j * nrow];
                printf("%d\t", tmp_int);
            }
            else
            {
                printf("%e\t", stat[i + j * nrow]);
            }
        }
        printf("\n");
    }
}

