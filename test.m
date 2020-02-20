 ## This demo:
 ##    - classifies a small set of unlabeled data points using
 ##      the Fuzzy C-Means algorithm into two fuzzy clusters
 ##    - plots the input points together with the cluster centers
 ##    - evaluates the quality of the resulting clusters using
 ##      three validity measures: the partition coefficient, the
 ##      partition entropy, and the Xie-Beni validity index
 ##
 ## Note: The input_data is taken from Chapter 13, Example 17 in
 ##       Fuzzy Logic: Intelligence, Control and Information, by
 ##       J. Yen and R. Langari, Prentice Hall, 1999, page 381
 ##       (International Edition). 

 ## Use fcm to classify the input_data.
 pkg load fuzzy-logic-toolkit 
 input_data = [2 12; 4 9; 7 13; 11 5; 12 7; 14 4];
 number_of_clusters = 2;
 [cluster_centers, soft_partition, obj_fcn_history] = fcm (input_data, number_of_clusters)
 
 ## Plot the data points as small blue x's.
 figure ('NumberTitle', 'off', 'Name', 'FCM Demo 1');
 for i = 1 : rows (input_data)
   plot (input_data(i, 1), input_data(i, 2), 'LineWidth', 2, 'marker', 'x', 'color', 'b');
   hold on;
 endfor

 ## Plot the cluster centers as larger red *'s.
 for i = 1 : number_of_clusters
   plot (cluster_centers(i, 1), cluster_centers(i, 2), 'LineWidth', 4, 'marker', '*', 'color', 'r');
   hold on;
 endfor

 ## Make the figure look a little better:
 ##    - scale and label the axes
 ##    - show gridlines
 xlim ([0 15]);
 ylim ([0 15]);
 xlabel ('Feature 1');
 ylabel ('Feature 2');
 grid
 hold
 
