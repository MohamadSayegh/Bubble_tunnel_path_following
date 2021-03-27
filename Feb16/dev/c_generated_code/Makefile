#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#


ACADOS_FLAGS = -fPIC -std=c99 #-fno-diagnostics-show-line-numbers -g
# # Debugging
# ACADOS_FLAGS += -g3

MODEL_OBJ=
MODEL_OBJ+= rockit_model_model/rockit_model_expl_ode_fun.o
MODEL_OBJ+= rockit_model_model/rockit_model_expl_vde_forw.o
MODEL_OBJ+= rockit_model_model/rockit_model_expl_ode_hess.o


OCP_OBJ=
OCP_OBJ+= rockit_model_constraints/rockit_model_constr_h_fun_jac_uxt_zt.o
OCP_OBJ+= rockit_model_constraints/rockit_model_constr_h_fun.o
OCP_OBJ+= rockit_model_constraints/rockit_model_constr_h_fun_jac_uxt_hess.o
OCP_OBJ+= rockit_model_constraints/rockit_model_constr_h_e_fun_jac_uxt_zt.o
OCP_OBJ+= rockit_model_constraints/rockit_model_constr_h_e_fun.o
OCP_OBJ+= rockit_model_constraints/rockit_model_constr_h_e_fun_jac_uxt_hess.o
OCP_OBJ+= rockit_model_cost/rockit_model_cost_ext_cost_fun.c
OCP_OBJ+= rockit_model_cost/rockit_model_cost_ext_cost_fun_jac.c
OCP_OBJ+= rockit_model_cost/rockit_model_cost_ext_cost_fun_jac_hess.c
OCP_OBJ+= rockit_model_cost/rockit_model_cost_ext_cost_e_fun.c
OCP_OBJ+= rockit_model_cost/rockit_model_cost_ext_cost_e_fun_jac.c
OCP_OBJ+= rockit_model_cost/rockit_model_cost_ext_cost_e_fun_jac_hess.c
OCP_OBJ+= acados_solver_rockit_model.o


SIM_OBJ=
SIM_OBJ+= acados_sim_solver_rockit_model.o

EX_OBJ=
EX_OBJ+= main_rockit_model.o

EX_SIM_OBJ=
EX_SIM_OBJ+= main_sim_rockit_model.o

OBJ=
OBJ+= $(MODEL_OBJ)
OBJ+= $(SIM_OBJ)
OBJ+= $(OCP_OBJ)

EXTERNAL_DIR=
EXTERNAL_LIB=

INCLUDE_PATH = /home/mohamad/acados/include
LIB_PATH = /home/mohamad/acados/lib
all: clean casadi_fun example_sim example
shared_lib: bundled_shared_lib ocp_shared_lib sim_shared_lib

CASADI_MODEL_SOURCE=
CASADI_MODEL_SOURCE+= rockit_model_expl_ode_fun.c
CASADI_MODEL_SOURCE+= rockit_model_expl_vde_forw.c
CASADI_MODEL_SOURCE+= rockit_model_expl_ode_hess.c
CASADI_CON_H_SOURCE=
CASADI_CON_H_SOURCE+= rockit_model_constr_h_fun_jac_uxt_zt.c
CASADI_CON_H_SOURCE+= rockit_model_constr_h_fun.c
CASADI_CON_H_SOURCE+= rockit_model_constr_h_fun_jac_uxt_hess.c
CASADI_CON_H_E_SOURCE=
CASADI_CON_H_E_SOURCE+= rockit_model_constr_h_e_fun_jac_uxt_zt.c
CASADI_CON_H_E_SOURCE+= rockit_model_constr_h_e_fun.c
CASADI_CON_H_E_SOURCE+= rockit_model_constr_h_e_fun_jac_uxt_hess.c


casadi_fun:
	( cd rockit_model_model; gcc $(ACADOS_FLAGS) -c  $(CASADI_MODEL_SOURCE))
	( cd rockit_model_constraints; gcc $(ACADOS_FLAGS) -c  $(CASADI_CON_H_SOURCE))
	( cd rockit_model_constraints; gcc $(ACADOS_FLAGS) -c  $(CASADI_CON_H_E_SOURCE))

main:
	gcc $(ACADOS_FLAGS) -c main_rockit_model.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/ \

main_sim:
	gcc $(ACADOS_FLAGS) -c main_sim_rockit_model.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/

ocp_solver:
	gcc $(ACADOS_FLAGS) -c acados_solver_rockit_model.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/ \

sim_solver:
	gcc $(ACADOS_FLAGS) -c acados_sim_solver_rockit_model.c -I $(INCLUDE_PATH)/blasfeo/include/ -I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) -I $(INCLUDE_PATH)/acados/  \

example: ocp_solver main
	gcc $(ACADOS_FLAGS) -o main_rockit_model $(EX_OBJ) $(OBJ) -L $(LIB_PATH) \
	-lacados -lhpipm -lblasfeo \
	-lm \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/hpipm/include/ \
	-I $(INCLUDE_PATH) \
	-I $(INCLUDE_PATH)/acados/ \


example_sim: sim_solver main_sim
	gcc $(ACADOS_FLAGS) -o main_sim_rockit_model $(EX_SIM_OBJ) $(MODEL_OBJ) $(SIM_OBJ) -L $(LIB_PATH) \
	-lacados -lhpipm -lblasfeo \
	-lm \
	-I $(INCLUDE_PATH)/blasfeo/include/ \
	-I $(INCLUDE_PATH)/acados/ \

bundled_shared_lib: casadi_fun ocp_solver sim_solver
	gcc $(ACADOS_FLAGS) -shared -o libacados_solver_rockit_model.so $(OBJ) \
	-L $(LIB_PATH) \
	-lacados -lhpipm -lblasfeo \
	-lm \

ocp_shared_lib: casadi_fun ocp_solver
	gcc $(ACADOS_FLAGS) -shared -o libacados_ocp_solver_rockit_model.so $(OCP_OBJ) $(MODEL_OBJ) \
	-L$(EXTERNAL_DIR) -l$(EXTERNAL_LIB) \
	-L $(LIB_PATH) -lacados -lhpipm -lblasfeo \
	-lm \

sim_shared_lib: casadi_fun sim_solver
	gcc $(ACADOS_FLAGS) -shared -o libacados_sim_solver_rockit_model.so $(SIM_OBJ) $(MODEL_OBJ) -L$(EXTERNAL_DIR) -l$(EXTERNAL_LIB)  \
	-L $(LIB_PATH) -lacados -lhpipm -lblasfeo \
	-lm \

clean:
	rm -f *.o
	rm -f *.so
	rm -f main_rockit_model

clean_ocp_shared_lib:
	rm -f libacados_ocp_solver_rockit_model.so
	rm -f acados_solver_rockit_model.o
