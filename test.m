clear all
close all
clc

N = 200;

theta = linspace(0,2*pi,N);
x1 = 3*(sin(theta) + 0.1*rand(1,N));
y1 = 3*(cos(theta) + 0.1*rand(1,N));  


theta2 = linspace(0,2*pi,N);
x2 = 6*(sin(theta2) + 0.1*rand(1,N))-1;
y2 = 6*(cos(theta2) + 0.1*rand(1,N))-1;  

sz = 4;

#scatter (x1, y1, sz,'r');
#hold on
#scatter (x2, y2, sz,'b');

X = [x1 x2]';
Y = [y1 y2]';

data = [X Y]';
figure(1)
plot(data(1,1:N),data(2,1:N),'mo')
hold on
plot(data(1,N+1:end),data(2,N+1:end),'gs')

#scatter (X, Y, sz,'g');
#hold on 

ep=[0.5 1.5 2.5];
sigma=[1 2 3];

figure(2)
count=1;
for k=1:length(ep)
    for z=1:length(sigma)
        subplot(3,3,count)
        hold on
        bel(k,z,:)=spectral_Ncut2(data,ep(k),sigma(z));
        for p=1:size(bel,3)
            if bel(k,z,p)==0
                plot(data(1,p),data(2,p),'mo')
            else
                plot(data(1,p),data(2,p),'gs')
            end
        end   
        titulo=sprintf('ep = %f || sigma = %f', ep(k), sigma(z));
        title(titulo);
        count=count+1;
    end
end