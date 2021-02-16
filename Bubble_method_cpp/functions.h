
#include <cstdio>
#include <cstdlib>
#include "spline.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>


#define _USE_MATH_DEFINES

#ifndef FUNCTION_H
#define FUNCTION_H


//----------------------------------------------------------------------------------------------------------------------------//
//                                                         Global Variables                                                                       //
//---------------------------------------------------------------------------------------------------------------------------//

tk::spline spline_g;


//----------------------------------------------------------------------------------------------------------------------------//
//                                                       Function Prototypes                                                                //
//---------------------------------------------------------------------------------------------------------------------------//

void                    generate_occupancy_grid(std::vector <double> &X, std::vector <double> &Y);
void                    generate_global_path(std::vector <double> &X, std::vector <double> &Y);
std::vector<double>     linspaced(double start, double end, int num);
std::vector<double>     constant_vector(double size,double num);
void                    defineDiscreteBubbles(std::vector <double> occupied_positions_x, std::vector <double> occupied_positions_y, std::vector <std::vector<double>>   &feasiblebubbles_x, std::vector <std::vector<double>>   &feasiblebubbles_y, std::vector<double> &midpoints_x, std::vector<double> &midpoints_y, std::vector <double> &radii);
double                  get_distance(double p1x,double p1y,double p2x,double p2y);
void                    multiply_scalars_vector(std::vector <double> &V, double a, double b);
void                    find_closest_point(double &smallest_distance, double &closest_x, double &closest_y, double point_x, double point_y, std::vector  <double>  set_x, std::vector <double> set_y);
void                    create_tunnel(std::vector <double> &interpolationpoints_x, std::vector <double> &interpolationpoints_y);
void                    NewBubbleGeneration(std::vector <double> global_path_x, std::vector <double> global_path_y, std::vector <double> occupied_positions_x, std::vector <double> occupied_positions_y, std::vector<double> &shifted_midpoints_x, std::vector<double> &shifted_midpoints_y, std::vector <double> &shifted_radii);


#endif