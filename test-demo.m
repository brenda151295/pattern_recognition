m1=[0;0];
m2=[10;0];
m3=[0;9];
m4=[9;8];
S1=eye(2);
S2=[1 0.2;0.2 1.5];
S3=[1 0.4;0.4 1.1];
S4=[0.3 0.2;0.2 0.5];

N=100;

randn('seed',0);

mat1=mvnrnd(m1,S1,N);
mat2=mvnrnd(m2,S2,N);
mat3=mvnrnd(m3,S3,N);
mat4=mvnrnd(m4,S4,N);
input_data=[mat1; mat2; mat3; mat4];

number_of_clusters = [4 3 5];
for cluster=number_of_clusters
   [cluster_centers, soft_partition, obj_fcn_history] = ...
     fcm (input_data, cluster, [NaN NaN NaN 0])

   ## Plot the data points in two dimensions (using features 1 & 2)
   ## as small blue x's.
   figure ('NumberTitle', 'off', 'Name', 'FCM Demo 2');
   for i = 1 : rows (input_data)
     plot (input_data(i, 1), input_data(i, 2), 'LineWidth', 2, ...
           'marker', 'x', 'color', 'b');
     hold on;
   endfor

   ## Plot the cluster centers in two dimensions
   ## (using features 1 & 2) as larger red *'s.
   for i = 1 : cluster
     plot (cluster_centers(i, 1), cluster_centers(i, 2), ...
           'LineWidth', 4, 'marker', '*', 'color', 'r');
     hold on;
   endfor

endfor


