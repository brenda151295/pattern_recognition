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

% distância entre pontos (1) = 2*r1/(N/2-1) = 6/99 = 0.0606
% distância entre pontos (2) = 2*r2/(N/2-1) = 12/99 = 0.1212

X=[];
for k=min1:0.0606:max1 
    val=sqrt(3^2 - k^2); 
    aleat1=(2*rand-1)/3; % +- 1/3
    aleat2=(2*rand-1)/3; % +- 1/3
    mat1=[k val+aleat1; k -val+aleat2];
    X=[X; mat1];
end
  
for k=min2:0.1212:max2 
    val=sqrt(6^2 - (k-1)^2); 
    aleat1=(2*rand-1)/3; % +- 1/3
    aleat2=(2*rand-1)/3; % +- 1/3
    mat2=[k val+aleat1+1; k -val+aleat2+1];
    X=[X; mat2];
end

X=X';

figure(1)
plot(X(1,1:N),X(2,1:N),'mo')
hold on
plot(X(1,N+1:end),X(2,N+1:end),'ks')

ep=[0.5 1.5 2.5];
sigma=[1 2 3];

figure(2)
count=1;
for k=1:length(ep)
    for z=1:length(sigma)
        subplot(3,3,count)
        hold on
        bel(k,z,:)=spectral_Ncut2(X,ep(k),sigma(z));
        for p=1:size(bel,3)
            if bel(k,z,p)==0
                plot(X(1,p),X(2,p),'mo')
            else
                plot(X(1,p),X(2,p),'ks')
            end
        end   
        titulo=sprintf('ep = %f || sigma = %f', ep(k), sigma(z));
        title(titulo);
        count=count+1;
    end
end

resp_1=sprintf('Observa-se que um "ep" muito pequeno, faz com que quase todos os pontos fiquem num mesmo grupo.\nJá um "ep" muito grande, faz com que alguns dados da segunda distribuição (no caso o círculo de raio menor) sejam alocados no mesmo grupo da primeira.\n Isto se deve ao fato que "ep" impacta no tamanho da "vizinhança".')
resp_2=sprintf('Já uma variação em "sigma" determina a largura do Kernel, impactando (no caso) nas parcelas dos circulos que são alocados em cada grupo.')

OBS=sprintf('A "figure 2" mostra os impactos das variações destes parâmetros.')