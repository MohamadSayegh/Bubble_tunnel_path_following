# Gnuplot script file for plotting data in file "forplot.txt"
# This file is called gnuplot_script.p
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xtic auto                          # set xtics automatically
set ytic auto                          # set ytics automatically
set title "Occupancy grid and global path"
set xlabel "X"
set ylabel "Y"
set key left top
set xr [0-2.0:12.0]
set yr [-2.0:15.0]
plot "occupancy_grid.txt" using 1:2 pt 7 ps 0.5  , "global_path.txt" with lines, "shifted_bubbles_midpoints.txt" with points , "shifted_bubbles.txt" using 1:2 pt 7 ps 0.5