clear all
close all
clc

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
X=[mat1; mat2; mat3; mat4]';

% A, B e C)

% F)

[thetaf1,Uf1,obj_funf1]=fuzzy_c_means(X,4,2);
[thetaf2,Uf2,obj_funf2]=fuzzy_c_means(X,3,2);
[thetaf3,Uf3,obj_funf3]=fuzzy_c_means(X,5,2);
Uf1=Uf1';
Uf2=Uf2';
Uf3=Uf3';
% Uf2=[Uf2; zeros(1,400);];
% Uf2=[Uf2; zeros(1,400); zeros(1,400)];
[val,belf(1,:)]=max(Uf1);
[val,belf(2,:)]=max(Uf2);
[val,belf(3,:)]=max(Uf3);
belf(4,:)=belf(1,:);

figure(1)
cores=['m*'; 'ko'; 'bv'; 'gp'; 'rs'];
titulo_fuzz=['FuzzyC | m=4 |'; 'FuzzyC | m=3 |'; 'FuzzyC | m=5 |'];
for p=1:5
    subplot(2,5,5+p)
    hold on
    if p<=3
        for k=1:size(belf,2)
            for z=1:max(belf(p,:))
                if belf(p,k)==z
                    plot(X(1,k),X(2,k),cores(z,:))
                end
            end
        end
        title(titulo_fuzz(p,:))
    else
        subplot(2,5,10)
        hold on
        plot(mat1(:,1),mat1(:,2),cores(1,:))
        plot(mat2(:,1),mat2(:,2),cores(2,:))
        plot(mat3(:,1),mat3(:,2),cores(3,:))
        plot(mat4(:,1),mat4(:,2),cores(4,:))
        title('Distribuições Originais')
    end 
end

clc

resp_a_f = sprintf('Ver figura 1 para os plots')
resp_g = sprintf('Considerando D1 = Distribuição com Média [0;0], D2 = [10;0], D3 = [0;9] e D4 = [9;8]:\nComentários K-means:\n -Com 4 centros aleatórios, foram criados exatamente um grupo para cada distribuição\n -Com 3 centros aleatórios, foram criados um grupo para D1, um para D3 e um para D2 e D4\n -Com 5 centros aleatórios, foi criado um grupo para D3, um para D2 e D4 e outros três grupos para D1\n -Com os 4 centros determinados, um grupo para D4, um para D2 e D3 (com excessão de um único dado de D3) e dois grupos para D1 (um deles comportando também o dado da D3 que ficou de fora)\n -Com três centros aleatórios e outro em [20 20], foram criados um grupo para D1, um para D4, um para D2 e D3 e um quarto grupo no qual não foi alocado nenhum dado\nComentários Fuzzy C-Means:\n -Com 4 centros, o resultado foi similar ao K-means com 4 centros aleatórios\n -Com 3 centros, foi criado um grupo para D1 e parte de D3, outro para D2 e a outra parte de D3 e outro para D4\n -Com cinco centros, um grupo para D1, outro para D2, um para D3 e dois para D4')
