clear all
close all
clc

N=200;

min1 = -3;
max1 = 3;
min2 = -6+1;
max2 = 6+1;

rand('seed',0);
randn('seed',0);


X=[];
for k=min1:0.0606:max1 
    val=sqrt(3^2 - k^2); 
    aleat1=(2*rand-1)/3; % +- 1/3
    aleat2=(2*rand-1)/3; % +- 1/3
    mat1=[k val+aleat1; k -val+aleat2];
    X=[X; mat1];
end
sprintf('K=')

  
for k=min2:0.1212:max2 
    val=sqrt(6^2 - (k-1)^2); 
    aleat1=(2*rand-1)/3; % +- 1/3
    aleat2=(2*rand-1)/3; % +- 1/3
    mat2=[k val+aleat1+1; k -val+aleat2+1];
    X=[X; mat2];
end

X=X';
X(1,1:N)