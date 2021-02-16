#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>

#include "functions.h"
#include "spline.h"

#define _USE_MATH_DEFINES
 
using namespace std;

//----------------------------------------------------------------------------------------------------------------------------//
//                                                      The functions                                                                                  //
//---------------------------------------------------------------------------------------------------------------------------//

void generate_occupancy_grid(vector <double> &Obstacles_positions_X, vector <double> &Obstacles_positions_Y){
 


    vector<double> new_Obstacle_positions_X;
    vector<double> new_Obstacle_positions_Y;

    // line at y = 10
    Obstacles_positions_X = linspaced(0.0, 9.0, 400);
    Obstacles_positions_Y = constant_vector(400, 10);    

    // line at x = 3
    new_Obstacle_positions_X = constant_vector(100, 3);  
    new_Obstacle_positions_Y = linspaced(0.0, 8.0, 100);

    Obstacles_positions_X.insert(Obstacles_positions_X.end(), new_Obstacle_positions_X.begin(), new_Obstacle_positions_X.end());
    Obstacles_positions_Y.insert(Obstacles_positions_Y.end(), new_Obstacle_positions_Y.begin(), new_Obstacle_positions_Y.end());

    // line at x = 7
    new_Obstacle_positions_X = constant_vector(100, 7);  
    new_Obstacle_positions_Y = linspaced(0.0, 8.0, 100);

    Obstacles_positions_X.insert(Obstacles_positions_X.end(), new_Obstacle_positions_X.begin(), new_Obstacle_positions_X.end());
    Obstacles_positions_Y.insert(Obstacles_positions_Y.end(), new_Obstacle_positions_Y.begin(), new_Obstacle_positions_Y.end());

   // line at y = 0
    new_Obstacle_positions_X = linspaced(3.0, 7.0, 400);  
    new_Obstacle_positions_Y = constant_vector(400, 0);  

    Obstacles_positions_X.insert(Obstacles_positions_X.end(), new_Obstacle_positions_X.begin(), new_Obstacle_positions_X.end());
    Obstacles_positions_Y.insert(Obstacles_positions_Y.end(), new_Obstacle_positions_Y.begin(), new_Obstacle_positions_Y.end());

   // line at y = 8
    new_Obstacle_positions_X = linspaced(7.0, 9.0, 100);  
    new_Obstacle_positions_Y = constant_vector(100, 8);  

    Obstacles_positions_X.insert(Obstacles_positions_X.end(), new_Obstacle_positions_X.begin(), new_Obstacle_positions_X.end());
    Obstacles_positions_Y.insert(Obstacles_positions_Y.end(), new_Obstacle_positions_Y.begin(), new_Obstacle_positions_Y.end());

    ofstream myfile1;
    myfile1.open("occupancy_grid.txt");
    for (int i=0; i<Obstacles_positions_X.size() ;i++)
        myfile1 << Obstacles_positions_X[i] << "   "<< Obstacles_positions_Y[i] << endl;
    myfile1.close();
 

}

void generate_global_path(vector <double> &xs, vector <double> &ys){
    

    // Define the global path that does not hit obstacles
    // here is still defined manually
    

    vector <double> X = linspaced(0,9,20);
    vector <double> Y = { 0.0, 5.0, 7.0, 8.0, 9.0, 9.0, 9.0, 7.0, 1.5, 1.5, 1.3, 1.3, 1.3, 1.5, 6.0, 9.0, 9.0, 9.0, 9.0, 9.0};
    
    //interpolate into spline
    spline_g.set_points(X,Y);

    double x = 0.0;
    while (x < 9.0){
        
        xs.push_back(x);
        ys.push_back(spline_g(x));
        x = x + 0.01;
   }
   
   ofstream myfile2;
   myfile2.open ("global_path.txt");
    for (int i=0; i< xs.size() ;i++)
        myfile2<< xs[i] << "   "<<ys[i]<<endl;
   myfile2.close();
    
}

vector<double> linspaced(double start, double end, int num){

vector<double> linspaced;

  if (num== 0) { 
      return linspaced; 
      }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  double delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); 
   
  return linspaced;
}

void defineDiscreteBubbles(vector <double> occupied_positions_x, vector <double> occupied_positions_y, vector <vector<double>>   &feasiblebubbles_x, vector <vector<double>>   &feasiblebubbles_y, vector<double> &midpoints_x, vector<double> &midpoints_y, vector <double> &radii){
        
        
    int len_occupied_positions = occupied_positions_x.size(); 
    double point_x = 0.0;
    double point_y = 0.0;
    int j = 0;
    int k = 0;
    double distance = 0;
    double smallest_distance =0;

    vector <double>  closest_points_x;
    vector <double>  closest_points_y;

    double max_radius = 5;
    double radius = 0;
    bool end_of_spline_reached = false;

    double delta_x = 0.0;
    double delta_y = 0.0;
    double midpoint_x = 0.0;
    double midpoint_y = 0.0;
    bool new_radius_feasible = false;

    vector <double> Ax;
    vector <double> Ay;
        
    // First two points
    double point1_x = 0.0;
    double point2_x = 0.1;   
    double point1_y = spline_g(point1_x);
    double point2_y = spline_g(point2_x);

    vector <double> ts;
    ts   = linspaced(0, 2*M_PI, 50);
    
    vector <double> costs;
    for(int q = 0 ; q<50;q++){
            costs.push_back(cos(ts[q]));
    }

    vector <double> sints;
    for(int q = 0; q<50;q++){
            sints.push_back(sin(ts[q]));
    }


    double dist = get_distance(point1_x ,point1_y ,point2_x ,point2_y );
    double ref_dist = ( point2_x -  point1_x)/dist;

    double closest_point_x;
    double closest_point_y;

    double x_step = 0.01;

    // ---------------------------------------------    While Loop ------------------------------------------------//
    
        
    while (! end_of_spline_reached){

        
        //Move over the spline until the end ==== since x until 10 only
        if (point_x >= 9.7){  
            // if the end is near
            point_x = 10;
            end_of_spline_reached = true;
        }
        point_y = spline_g(point_x); 
        
        
        find_closest_point(smallest_distance, closest_point_x, closest_point_y, point_x, point_y, occupied_positions_x, occupied_positions_y);
        closest_points_x.push_back(closest_point_x); 
        closest_points_x.push_back(closest_point_y);


        
        if (smallest_distance < max_radius){
            
            delta_x = point_x - occupied_positions_x[k];
            delta_y = point_y - occupied_positions_y[k];

            midpoint_x = point_x + (1/2)*delta_x;
            midpoint_y = point_y + (1/2)*delta_y;

            radius = smallest_distance;   // factor 2.9/2 wa multiplied here to increase radius ?
            new_radius_feasible = true;
    
            j = 0;
            while (j <len_occupied_positions){
                    
                    distance = get_distance(midpoint_x,midpoint_y,occupied_positions_x[j],occupied_positions_y[j]);
                
                    if (distance >= radius){
                        j = j + 1;
                    }
                
                    else {
                        new_radius_feasible = false;
                        j = len_occupied_positions;
                    }
            }
            
        
            if (new_radius_feasible){
                    
                radii.push_back(radius);
                midpoints_x.push_back(midpoint_x);
                midpoints_y.push_back(midpoint_y);
                smallest_distance = radius;
            
            }
            
            else{
                radii.push_back(smallest_distance);
                midpoints_x.push_back(point_x);
                midpoints_y.push_back(point_y);
                midpoint_x = point_x;
                midpoint_y = point_y;
                radius = smallest_distance;
            }
        }
        
        
        // if  not (smallest_distance < max_radius)
        else {    
            radii.push_back(smallest_distance);
            midpoints_x.push_back(point_x);
            midpoints_y.push_back(point_y);
            midpoint_x = point_x;
            midpoint_y = point_y;
            radius = smallest_distance;
            
        }
                
        // calculate the vecotrs to add to feasible bubbles
        Ax = costs;
        Ay = sints;
        multiply_scalars_vector(Ax,  midpoint_x, radius);
        multiply_scalars_vector(Ay,  midpoint_y, radius);
        feasiblebubbles_x.push_back(Ax);          
        feasiblebubbles_y.push_back(Ay);

        if (radius >= 4){            // why 4 ?
            point_x = point_x + ref_dist*radius;
        }
        else if (ref_dist*radius <= 0.10){
            point_x = point_x + ref_dist*radius;
        }
        else{
            point_x = point_x + x_step;   // this controls the size of jump (how many times to draw the bubbles
        }
        j = 0;

    } // end of while loop


    // End of  function
}

double get_distance(double p1x,double p1y,double p2x,double p2y){
    
   double dist;
   dist =  pow((p1x- p2x),2) + pow((p1y - p2y),2);
//    dist  = pow(dist,0.5);   //to reduce computations
   return dist; 

}

void multiply_scalars_vector(vector <double> &V, double a, double b){
    
    for(unsigned int i = 0; i<V.size();i++){
        V[i] = a + b*V[i];
    }
}

void defineInterpolationLists(vector<double> &midpoints_x, vector<double> &midpoints_y, vector <double> &radii,vector <double> &interpolationpoints_x, vector <double> &interpolationpoints_y){
        
        
    int index = 0;
    vector <double> interpolationradii; 
    interpolationpoints_x.push_back(midpoints_x[0]);
    interpolationpoints_y.push_back(midpoints_y[0]);
    interpolationradii.push_back(radii[0]);

    double midpoint_1_x,midpoint_1_y;
    double midpoint_2_x, midpoint_2_y;
    double radius_1, radius_2;
    double dist;
    double helpingpoint_x, helpingpoint_y;
    double intersection_1_x, intersection_1_y;
    double intersection_2_x, intersection_2_y;
    double radius_intersection;
    double dist1,dist2,dist3;

    bool intersections_present = true;
    int len = midpoints_x.size() - 1;

    double a = 0;
    double h = 0;
    
    while(index < len) {
        midpoint_1_x = midpoints_x[index];
        midpoint_1_y = midpoints_y[index];
        radius_1 = radii[index];
        midpoint_2_x = midpoints_x[index +1];
        midpoint_2_y = midpoints_y[index + 1];
        radius_2 = radii[index + 1];
        
        dist = get_distance(midpoint_1_x,midpoint_1_y,midpoint_2_x,midpoint_2_y);
        
        if(dist > radius_1 + radius_2){
            cout<< "These circles are not intersecting"<<endl;
            intersections_present = false;
        }
        else if(dist < abs(radius_1 - radius_2)){
            cout<< "One of the circles is in another circle"<<endl;
            intersections_present = false;
        }
        else if(dist  == 0 &&  (radius_1 == radius_2)){
            cout<< "One of the circles is in another circle"<<endl;
        intersections_present = false;
        }
        else{ //if no other condition happens then there is an intersection
        
            intersections_present = true;
            
            a =   ( pow(radius_1,2)   -   pow(radius_2,2)  ) + (pow(dist,2)/(2*dist));
            h =    pow((pow(radius_1,2) - pow(a,2)),0.5);
            
            helpingpoint_x = midpoint_1_x + (a*(midpoint_2_x - midpoint_1_x)/dist);
            helpingpoint_y = midpoint_1_y + (a*(midpoint_2_y - midpoint_1_y)/dist);
            
            intersection_1_x = helpingpoint_x + (h*(midpoint_2_y - midpoint_1_y)/dist);
            intersection_1_y = helpingpoint_y -  (h*(midpoint_2_x- midpoint_1_x)/dist);
            
            intersection_2_x = helpingpoint_x - (h*(midpoint_2_y - midpoint_1_y)/dist);
            intersection_2_y = helpingpoint_y + (h*(midpoint_2_x- midpoint_1_x)/dist);
            
            radius_intersection = pow(  (helpingpoint_x - pow(intersection_1_x,2)   ) + (helpingpoint_y - pow(intersection_1_y,2)   ),0.5   );
            
            
            //List the radii and the interpolation points
            dist1 = pow(pow((midpoint_1_x - helpingpoint_x),2) + pow((midpoint_1_y - helpingpoint_y),2),0.5);
            dist2 = pow(pow((midpoint_2_x - helpingpoint_x),2) + pow((midpoint_2_y - helpingpoint_y),2),0.5);
            dist3 = pow(pow((midpoint_1_x - midpoint_2_x),2) + pow((midpoint_1_y - midpoint_2_y),2),0.5);
            
    //        cout<<"dist1  "<<dist1<<"  dist2  "<<dist2<<"  dist3  "<<dist3<<endl;
            
            if ((dist1 + dist2) == dist3){   //Helping point in between the two midpoints
                interpolationpoints_x.push_back(helpingpoint_x);
                interpolationpoints_y.push_back(helpingpoint_y);
                interpolationradii.push_back(radius_intersection);
                interpolationpoints_x.push_back(midpoint_2_x);
                interpolationpoints_y.push_back(midpoint_2_y);
                interpolationradii.push_back(radius_2);
            }
            else{   // Helping point not in between the two midpoints
                interpolationpoints_x.push_back(midpoint_2_x);
                interpolationpoints_y.push_back(midpoint_2_y);
                interpolationradii.push_back(radius_2);
            }


        } 
        index = index +1;
        }
    

    // ---------------------------------------------     Generate data for plots ------------------------------------------------//

    // 1) for Plotting the interpolation, discrete midpoints and the occupied positions
    // Be careful there are midpoint(s) and a variable called midpoint!

   ofstream myfile5;
   myfile5.open ("interpolationpoints.txt");
    for (unsigned int i=0; i<interpolationpoints_x.size();i++){
        myfile5<< interpolationpoints_x[i]<<"   "<<interpolationpoints_y[i]<<endl;
    }
   myfile5.close();
   

// End of function

   
}
 
void create_tunnel(vector <double> &interpolationpoints_x, vector <double> &interpolationpoints_y){
        
    int index = 1;
    int dist = 0;
    double refpoint_x = interpolationpoints_x[0];
    double refpoint_y = interpolationpoints_y[0];
    double currentpoint_x;
    double currentpoint_y;
    vector <double> svar ;
    svar.push_back(dist);
    
    //Calculate the path variable for every interpolation point
    //This path variable is the straight distance while following
    //the different interpolation points

        int len = interpolationpoints_x.size();

    while (index < len){
        currentpoint_x = interpolationpoints_x[index];
        currentpoint_y = interpolationpoints_y[index];

        dist = dist + get_distance(currentpoint_x,currentpoint_y,refpoint_x,refpoint_y);

        svar.push_back(dist);

        refpoint_x = currentpoint_x;
        refpoint_y = currentpoint_y;
        index = index + 1;
    }

}

void find_closest_point(double &smallest_distance, double &closest_x, double &closest_y, double point_x, double point_y, vector  <double>  set_x, vector <double> set_y){
    
    double len = set_x.size();
    double dist = 0;
    int k = 0;
    
     for(int j = 0; j < len; j++) {

            dist= get_distance(point_x,point_y,set_x[j],set_y[j]);

            if (j == 0){
                smallest_distance = dist;
                k = j;
            }
            if (dist < smallest_distance){
                smallest_distance = dist;
                k = j;
            }
    }
        
            closest_x = set_x[k]; 
            closest_y = set_y[k];
            
} 

vector<double> constant_vector(double size,double num){

vector<double> vect;

for(int i = 0; i< size ;i++){
    vect.push_back(num); 
}

return vect;

}


void NewBubbleGeneration(vector <double> global_path_x, vector <double> global_path_y, vector <double> occupied_positions_x, vector <double> occupied_positions_y, vector<double> &shifted_midpoints_x, vector<double> &shifted_midpoints_y, vector <double> &shifted_radii){



    double point_x = 0.0;
    double point_y = 0.0; 

    double new_point_x = 0.0;
    double new_point_y = 0.0;

    double new_shifted_point_x = 0.0;
    double new_shifted_point_y = 0.0;

    double distance    = 0.0;
    double radius      = 0.0;
    double new_radius  = 0.0;

    double delta_x = 0.0;
    double delta_y = 0.0;


    bool midpoint_feasible = false;
    bool new_point_inside_bubble = true;
    bool obstacle_inside = false;


    double shifted_radius  = 0.0;
    double shifted_point_x = 0.0;
    double shifted_point_y = 0.0;

    double acceptable_radius = 2.0;

    double closest_point_x = 0.0;
    double closest_point_y = 0.0;

    int index  = 0;
    int indexp = 0;

    double path_length = global_path_x.size();


    // ---------------------------------------------    While Loop ------------------------------------------------//
          
    while(index < path_length){

        point_x = global_path_x[index];
        point_y = global_path_y[index];

 
        // for choosing the bubble radius
        find_closest_point(radius, closest_point_x, closest_point_y, point_x, point_y, occupied_positions_x, occupied_positions_y);
        radius = pow(radius, 0.5); //commented in get_distance


        // radius = get_distance(point_x,point_y,closest_point_x,closest_point_y);

        radius = 0.9*radius ;  //for safety 

        // for choosing next midpoint
        new_point_inside_bubble = true;
        indexp = index;

        while(new_point_inside_bubble == true) {

            indexp = indexp + 1;

            if (indexp >= path_length){
                index = path_length;
                break;
            }

            new_point_x = global_path_x[indexp];
            new_point_y = global_path_y[indexp];

            distance = get_distance(new_point_x,new_point_y,point_x,point_y);
            distance = pow(distance, 0.5);


            if (distance >= radius){

                new_point_inside_bubble = false;
                index = indexp;

            }

        }


        // for shifting the midpoints
        shifted_radius   = radius;
        shifted_point_x  = point_x;
        shifted_point_y  = point_y;
        new_radius       = 0.0;

        if (radius < acceptable_radius){

            delta_x = point_x - closest_point_x;
            delta_y = point_y - closest_point_y;

            new_shifted_point_x = point_x + delta_x;
            new_shifted_point_y = point_y + delta_y;

            find_closest_point(new_radius, closest_point_x, closest_point_y, new_shifted_point_x, new_shifted_point_y, occupied_positions_x, occupied_positions_y);
            new_radius = pow(new_radius, 0.5);

            if (new_radius >= radius){
                shifted_radius  = new_radius;
                shifted_point_x = new_shifted_point_x;
                shifted_point_y = new_shifted_point_y;
            }
            
        }


        shifted_midpoints_x.push_back(shifted_point_x);
        shifted_midpoints_y.push_back(shifted_point_y);
        shifted_radii.push_back(shifted_radius);

        
    } // end of while loop


    // End of  function
}



