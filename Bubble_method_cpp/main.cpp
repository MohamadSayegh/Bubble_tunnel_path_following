#include <cstdio>
#include <cstdlib>

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

#include "functions.h"
#include "spline.h"
#include "time.h"

#define _USE_MATH_DEFINES
 
using namespace std;




//----------------------------------------------------------------------------------------------------------------------------//
//                                                                        Main                                                //
//---------------------------------------------------------------------------------------------------------------------------//

int main()
{
    
   //  Obstacle points
   vector<double> Obstacles_positions_X ;
   vector<double> Obstacles_positions_Y;
 
   // x linespace into 100 parts
   vector<double> global_path_X;
   vector<double> global_path_Y;

   generate_occupancy_grid(Obstacles_positions_X ,Obstacles_positions_Y);
    
   generate_global_path(global_path_X , global_path_Y );
   
   //for the bubbles
   vector <vector<double>>  shifted_feasiblebubbles_x ;
   vector <vector<double>>  shifted_feasiblebubbles_y ;
   vector <double>  shifted_midpoints_x ;
   vector <double>  shifted_midpoints_y ;
   vector <double>  shifted_radii;


   clock_t start, end; 
  
   /* Recording the starting clock tick.*/
   start = clock(); 


   // old method === needs to initialize vectors to be called
   // defineDiscreteBubbles(Obstacles_positions_X , Obstacles_positions_Y, feasiblebubbles_x , feasiblebubbles_y, midpoints_x, midpoints_y,radii);
    
   //new method
   NewBubbleGeneration(global_path_X, global_path_Y, Obstacles_positions_X , Obstacles_positions_Y, shifted_midpoints_x, shifted_midpoints_y,shifted_radii);
 
   // Calculating total time taken by the program. 
   end = clock(); 
  
   // Calculating total time taken by the program. 
   double time_taken = double(end - start) / double(CLOCKS_PER_SEC); 
   cout << "Time taken by program is : " << time_taken << " sec " << endl; 

    // create feasible bubbles points

    vector <double> ts;
    ts   = linspaced(0, 2*M_PI, 100);
    
    vector <double> costs;
    for(int q = 0 ; q<100;q++){
            costs.push_back(cos(ts[q]));
    }

    vector <double> sints;
    for(int q = 0; q<100;q++){
            sints.push_back(sin(ts[q]));
    }


    vector <double> Ax = costs;
    vector <double> Ay = sints;
    
    for (int i = 0; i < shifted_midpoints_x.size();i++){
        
        Ax = costs;
        Ay = sints;
        multiply_scalars_vector(Ax,  shifted_midpoints_x[i], shifted_radii[i]);
        multiply_scalars_vector(Ay,  shifted_midpoints_y[i], shifted_radii[i]);
        shifted_feasiblebubbles_x.push_back(Ax);          
        shifted_feasiblebubbles_y.push_back(Ay);

    }

   // ---------------------------------------------     Generate data for plots ------------------------------------------------//
 

   ofstream myfile5;
   myfile5.open ("shifted_bubbles_midpoints.txt");
    for (unsigned int i=0; i<shifted_midpoints_x.size();i++){
        myfile5<< shifted_midpoints_x[i]<<"   "<<shifted_midpoints_y[i]<<endl;
    }
   myfile5.close();
   

   ofstream myfile6;
   myfile6.open ("shifted_bubbles.txt");
    for (unsigned int i=0; i<shifted_feasiblebubbles_x.size();i++){
        for(unsigned int j = 0; j<shifted_feasiblebubbles_x[i].size();j++){
            myfile6<< shifted_feasiblebubbles_x[i][j]<<"   "<<shifted_feasiblebubbles_y[i][j]<<endl;
        }
    }
   myfile6.close();
   
   return 0;


}

