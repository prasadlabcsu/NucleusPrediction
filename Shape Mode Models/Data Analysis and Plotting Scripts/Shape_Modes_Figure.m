%% Import PCA and SM Data

clear all
close all force
clc

load('PCAMatrices.mat')
load('SMStats.mat')

%% SM Vectors

% Cell

SM_Cell_0 = zeros(20,1);
SM_Cell_p2 = zeros(20,20);
SM_Cell_n2 = zeros(20,20);
for k = 1:20
    SM_Cell_p2(k,k) = 2;
    SM_Cell_n2(k,k) = -2;
end

% Nucleus

SM_Nuc_0 = zeros(12,1);
SM_Nuc_p2 = zeros(12,12);
SM_Nuc_n2 = zeros(12,12);
for k = 1:12
    SM_Nuc_p2(k,k) = 2;
    SM_Nuc_n2(k,k) = -2;
end

%% SM to SHE

% Reverse SM z-scoring

    % Cell

for k = 1:20
    SM_Cell_0(k) = SM_Cell_0(k)*SM_cell_Inter_std(k) + SM_cell_Inter_mean(k);
    SM_Cell_p2(:,k) = SM_Cell_p2(:,k).*SM_cell_Inter_std' + SM_cell_Inter_mean';
    SM_Cell_n2(:,k) = SM_Cell_n2(:,k).*SM_cell_Inter_std' + SM_cell_Inter_mean';
end

    % Nucleus

for k = 1:12
    SM_Nuc_0(k) = SM_Nuc_0(k)*SM_nuc_Inter_std(k) + SM_nuc_Inter_mean(k);
    SM_Nuc_p2(:,k) = SM_Nuc_p2(:,k).*SM_nuc_Inter_std' + SM_nuc_Inter_mean';
    SM_Nuc_n2(:,k) = SM_Nuc_n2(:,k).*SM_nuc_Inter_std' + SM_nuc_Inter_mean';
end

% Reverse PCA

    % Cell

SHE_Cell_0 = (SM_Cell_0'*PCA_cell_Inter(:,1:20)')';

SHE_Cell_p2 = zeros(289,20);
SHE_Cell_n2 = zeros(289,20);
for k = 1:20
    SHE_Cell_p2(:,k) = (SM_Cell_p2(:,k)'*PCA_cell_Inter(:,1:20)')';
    SHE_Cell_n2(:,k) = (SM_Cell_n2(:,k)'*PCA_cell_Inter(:,1:20)')';
end

    % Nucleus

SHE_Nuc_0 = (SM_Nuc_0'*PCA_nuc_Inter(:,1:12)')';

SHE_Nuc_p2 = zeros(289,12);
SHE_Nuc_n2 = zeros(289,12);
for k = 1:12
    SHE_Nuc_p2(:,k) = (SM_Nuc_p2(:,k)'*PCA_nuc_Inter(:,1:12)')';
    SHE_Nuc_n2(:,k) = (SM_Nuc_n2(:,k)'*PCA_nuc_Inter(:,1:12)')';
end

%% SHE to SH to XYZ

% SHE to XYZ

Resolution = 100;

    % Cell

[SHE_Cell_0_Coef_C,SHE_Cell_0_Coef_S] = SHE_Formatter(SHE_Cell_0);
[Cell_0_X,Cell_0_Y,Cell_0_Z] = SHE_2_Cart(SHE_Cell_0_Coef_C,SHE_Cell_0_Coef_S,Resolution);
Cell_0_X = 0.108333.*Cell_0_X;
Cell_0_Y = 0.108333.*Cell_0_Y;
Cell_0_Z = 0.108333.*Cell_0_Z;

Cell_p2_X = zeros(Resolution,Resolution,20);
Cell_p2_Y = zeros(Resolution,Resolution,20);
Cell_p2_Z = zeros(Resolution,Resolution,20);
Cell_n2_X = zeros(Resolution,Resolution,20);
Cell_n2_Y = zeros(Resolution,Resolution,20);
Cell_n2_Z = zeros(Resolution,Resolution,20);
for k = 1:20
    [SHE_Cell_p2_Coef_C,SHE_Cell_p2_Coef_S] = SHE_Formatter(SHE_Cell_p2(:,k));
    [Cell_p2_X_temp,Cell_p2_Y_temp,Cell_p2_Z_temp] = SHE_2_Cart(SHE_Cell_p2_Coef_C,SHE_Cell_p2_Coef_S,Resolution);
    Cell_p2_X(:,:,k) = 0.108333.*Cell_p2_X_temp;
    Cell_p2_Y(:,:,k) = 0.108333.*Cell_p2_Y_temp;
    Cell_p2_Z(:,:,k) = 0.108333.*Cell_p2_Z_temp;

    [SHE_Cell_n2_Coef_C,SHE_Cell_n2_Coef_S] = SHE_Formatter(SHE_Cell_n2(:,k));
    [Cell_n2_X_temp,Cell_n2_Y_temp,Cell_n2_Z_temp] = SHE_2_Cart(SHE_Cell_n2_Coef_C,SHE_Cell_n2_Coef_S,Resolution);
    Cell_n2_X(:,:,k) = 0.108333.*Cell_n2_X_temp;
    Cell_n2_Y(:,:,k) = 0.108333.*Cell_n2_Y_temp;
    Cell_n2_Z(:,:,k) = 0.108333.*Cell_n2_Z_temp;
end

    % Nucleus

[SHE_Nuc_0_Coef_C,SHE_Nuc_0_Coef_S] = SHE_Formatter(SHE_Nuc_0);
[Nuc_0_X,Nuc_0_Y,Nuc_0_Z] = SHE_2_Cart(SHE_Nuc_0_Coef_C,SHE_Nuc_0_Coef_S,Resolution);
Nuc_0_X = 0.108333.*Nuc_0_X;
Nuc_0_Y = 0.108333.*Nuc_0_Y;
Nuc_0_Z = 0.108333.*Nuc_0_Z;

Nuc_p2_X = zeros(Resolution,Resolution,20);
Nuc_p2_Y = zeros(Resolution,Resolution,20);
Nuc_p2_Z = zeros(Resolution,Resolution,20);
Nuc_n2_X = zeros(Resolution,Resolution,20);
Nuc_n2_Y = zeros(Resolution,Resolution,20);
Nuc_n2_Z = zeros(Resolution,Resolution,20);
for k = 1:12
    [SHE_Nuc_p2_Coef_C,SHE_Nuc_p2_Coef_S] = SHE_Formatter(SHE_Nuc_p2(:,k));
    [Nuc_p2_X_temp,Nuc_p2_Y_temp,Nuc_p2_Z_temp] = SHE_2_Cart(SHE_Nuc_p2_Coef_C,SHE_Nuc_p2_Coef_S,Resolution);
    Nuc_p2_X(:,:,k) = 0.108333.*Nuc_p2_X_temp;
    Nuc_p2_Y(:,:,k) = 0.108333.*Nuc_p2_Y_temp;
    Nuc_p2_Z(:,:,k) = 0.108333.*Nuc_p2_Z_temp;

    [SHE_Nuc_n2_Coef_C,SHE_Nuc_n2_Coef_S] = SHE_Formatter(SHE_Nuc_n2(:,k));
    [Nuc_n2_X_temp,Nuc_n2_Y_temp,Nuc_n2_Z_temp] = SHE_2_Cart(SHE_Nuc_n2_Coef_C,SHE_Nuc_n2_Coef_S,Resolution);
    Nuc_n2_X(:,:,k) = 0.108333.*Nuc_n2_X_temp;
    Nuc_n2_Y(:,:,k) = 0.108333.*Nuc_n2_Y_temp;
    Nuc_n2_Z(:,:,k) = 0.108333.*Nuc_n2_Z_temp;
end

%% Plotting

% Average cell and nucleus mesh shape plot

figure
surf(Cell_0_X,Cell_0_Y,Cell_0_Z,'EdgeColor','m','FaceColor','none')
hold on
surf(Nuc_0_X,Nuc_0_Y,Nuc_0_Z,'EdgeColor','b','FaceColor','none')
title('Average Cell and Nucleus (z = 0)')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca; 
ax.FontSize = 36;
hold off

% Average cell mesh shape plot for 3 views

figure

subplot(1,3,1)
surf(Cell_0_X,Cell_0_Y,Cell_0_Z,'EdgeColor','m','FaceColor','none')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
view(0,0)
set(gca,'DataAspectRatio',[1 1 1])

subplot(1,3,2)
surf(Cell_0_X,Cell_0_Y,Cell_0_Z,'EdgeColor','m','FaceColor','none')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
view(90,0)
set(gca,'DataAspectRatio',[1 1 1])

subplot(1,3,3)
surf(Cell_0_X,Cell_0_Y,Cell_0_Z,'EdgeColor','m','FaceColor','none')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
view(0,90)
set(gca,'DataAspectRatio',[1 1 1])

% Average nuclear mesh shape plot for 3 views

figure

subplot(1,3,1)
surf(Nuc_0_X,Nuc_0_Y,Nuc_0_Z,'EdgeColor','b','FaceColor','none')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
view(0,0)
set(gca,'DataAspectRatio',[1 1 1])

subplot(1,3,2)
surf(Nuc_0_X,Nuc_0_Y,Nuc_0_Z,'EdgeColor','b','FaceColor','none')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
view(90,0)
set(gca,'DataAspectRatio',[1 1 1])

subplot(1,3,3)
surf(Nuc_0_X,Nuc_0_Y,Nuc_0_Z,'EdgeColor','b','FaceColor','none')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
view(0,90)
set(gca,'DataAspectRatio',[1 1 1])

% Positive and negative extremes cell plots

% for k = 1:20
% 
%     name_str = strcat('Cell, Shape Mode:',' ',num2str(k),', Postive and Negative Extremes');
%     figure
%     surf(Cell_p2_X(:,:,k),Cell_p2_Y(:,:,k),Cell_p2_Z(:,:,k),'EdgeColor','r','FaceColor','none')
%     hold on
%     surf(Cell_n2_X(:,:,k),Cell_n2_Y(:,:,k),Cell_n2_Z(:,:,k),'EdgeColor','b','FaceColor','none')
%     title(name_str)
%     xlabel('X (\mum)')
%     ylabel('Y (\mum)')
%     zlabel('Z (\mum)')
%     ax = gca; 
%     ax.FontSize = 36;
% 
% end

% List of the most illustrative planes for each SM

    % SM 1: XZ view(0,0)
    % SM 2: XZ view(0,0)
    % SM 3: XZ view(0,0)
    % SM 4: XZ view(0,0)
    % SM 5: XY view(0,90)
    % SM 6: XY view(0,90)
    % SM 7: XY view(0,90)
    % SM 8: XZ view(0,0)
    % SM 9: XY view(0,90)
    % SM 10: XY view(0,90)
    % SM 11: YZ view(90,0)
    % SM 12: YZ view(90,0)
    % SM 13: XZ view(0,0)
    % SM 14: XY view(0,90)
    % SM 15: XZ view(0,0)
    % SM 16: XZ view(0,0)
    % SM 17: XZ view(0,0)
    % SM 18: XZ view(0,0)
    % SM 19: XZ view(0,0)
    % SM 20: XZ view(0,0)

viewX = [0,0,0,0,0,0,0,0,0,0,90,90,0,0,0,0,0,0,0,0];
viewY = [0,0,0,0,90,90,90,0,90,90,0,0,0,90,0,0,0,0,0,0];

% All cell shape modes plot

figure

for k = 1:20
    
    name_str = strcat('SM',num2str(k));
    subplot(4,5,k)
    surf(Cell_p2_X(:,:,k),Cell_p2_Y(:,:,k),Cell_p2_Z(:,:,k),'EdgeColor','r','FaceColor','none')
    hold on
    surf(Cell_n2_X(:,:,k),Cell_n2_Y(:,:,k),Cell_n2_Z(:,:,k),'EdgeColor','b','FaceColor','none')
    title(name_str)
    xlabel('X (\mum)')
    ylabel('Y (\mum)')
    zlabel('Z (\mum)')
    set(gca,'DataAspectRatio',[1 1 1])
    view(viewX(k),viewY(k))
    hold off
    grid off

end

% Postive and negative extremes nucleus plots

% for k = 1:12
% 
%     name_str = strcat('Nucleus, Shape Mode:',' ',num2str(k),', Postive and Negative Extremes');
%     figure
%     surf(Nuc_p2_X(:,:,k),Nuc_p2_Y(:,:,k),Nuc_p2_Z(:,:,k),'EdgeColor','r','FaceColor','none')
%     hold on
%     surf(Nuc_n2_X(:,:,k),Nuc_n2_Y(:,:,k),Nuc_n2_Z(:,:,k),'EdgeColor','b','FaceColor','none')
%     title(name_str)
%     xlabel('X (\mum)')
%     ylabel('Y (\mum)')
%     zlabel('Z (\mum)')
%     ax = gca; 
%     ax.FontSize = 36;
% 
% end

% List of the most illustrative planes for each SM

    % SM 1: XZ view(0,0)
    % SM 2: YZ view(90,0)
    % SM 3: XY view(0,90)
    % SM 4: XZ view(0,0)
    % SM 5: XZ view(0,0)
    % SM 6: XY view(0,90)
    % SM 7: XY view(0,90)
    % SM 8: YZ view(90,0)
    % SM 9: XY view(0,90)
    % SM 10: XZ view(0,0)
    % SM 11: XZ view(0,0)
    % SM 12: XZ view(0,0) 

viewX = [0,90,0,0,0,0,0,90,0,0,0,0];
viewY = [0,0,90,0,0,90,90,0,90,0,0,0];

% All nucleus shape modes plot

figure

for k = 1:12
    
    name_str = strcat('SM',num2str(k));
    subplot(3,4,k)
    surf(Nuc_p2_X(:,:,k),Nuc_p2_Y(:,:,k),Nuc_p2_Z(:,:,k),'EdgeColor','r','FaceColor','none')
    hold on
    surf(Nuc_n2_X(:,:,k),Nuc_n2_Y(:,:,k),Nuc_n2_Z(:,:,k),'EdgeColor','b','FaceColor','none')
    title(name_str)
    xlabel('X (\mum)')
    ylabel('Y (\mum)')
    zlabel('Z (\mum)')
    set(gca,'DataAspectRatio',[1 1 1])
    view(viewX(k),viewY(k))
    hold off
    grid off

end

%% Individual Plots

for k = 1:20

    figure

    subplot(1,3,1)
    name_str = strcat('SM',num2str(k));
    surf(Cell_p2_X(:,:,k),Cell_p2_Y(:,:,k),Cell_p2_Z(:,:,k),'EdgeColor','r','FaceColor','none')
    hold on
    surf(Cell_n2_X(:,:,k),Cell_n2_Y(:,:,k),Cell_n2_Z(:,:,k),'EdgeColor','b','FaceColor','none')
    title(name_str)
    xlabel('X (\mum)')
    ylabel('Y (\mum)')
    zlabel('Z (\mum)')
    set(gca,'DataAspectRatio',[1 1 1])
    hold off
    grid off
    view(0,0)
    
    subplot(1,3,2)
    surf(Cell_0_X,Cell_0_Y,Cell_0_Z,'EdgeColor','m','FaceColor','none')
    name_str = strcat('SM',num2str(k));
    surf(Cell_p2_X(:,:,k),Cell_p2_Y(:,:,k),Cell_p2_Z(:,:,k),'EdgeColor','r','FaceColor','none')
    hold on
    surf(Cell_n2_X(:,:,k),Cell_n2_Y(:,:,k),Cell_n2_Z(:,:,k),'EdgeColor','b','FaceColor','none')
    title(name_str)
    xlabel('X (\mum)')
    ylabel('Y (\mum)')
    zlabel('Z (\mum)')
    set(gca,'DataAspectRatio',[1 1 1])
    hold off
    grid off
    view(90,0)
    
    subplot(1,3,3)
    name_str = strcat('SM',num2str(k));
    surf(Cell_p2_X(:,:,k),Cell_p2_Y(:,:,k),Cell_p2_Z(:,:,k),'EdgeColor','r','FaceColor','none')
    hold on
    surf(Cell_n2_X(:,:,k),Cell_n2_Y(:,:,k),Cell_n2_Z(:,:,k),'EdgeColor','b','FaceColor','none')
    title(name_str)
    xlabel('X (\mum)')
    ylabel('Y (\mum)')
    zlabel('Z (\mum)')
    set(gca,'DataAspectRatio',[1 1 1])
    hold off
    grid off
    view(0,90)

end

for k = 1:12

    figure

    subplot(1,3,1)
    name_str = strcat('SM',num2str(k));
    surf(Nuc_p2_X(:,:,k),Nuc_p2_Y(:,:,k),Nuc_p2_Z(:,:,k),'EdgeColor','r','FaceColor','none')
    hold on
    surf(Nuc_n2_X(:,:,k),Nuc_n2_Y(:,:,k),Nuc_n2_Z(:,:,k),'EdgeColor','b','FaceColor','none')
    title(name_str)
    xlabel('X (\mum)')
    ylabel('Y (\mum)')
    zlabel('Z (\mum)')
    set(gca,'DataAspectRatio',[1 1 1])
    hold off
    grid off
    view(0,0)
    
    subplot(1,3,2)
    name_str = strcat('SM',num2str(k));
    surf(Nuc_p2_X(:,:,k),Nuc_p2_Y(:,:,k),Nuc_p2_Z(:,:,k),'EdgeColor','r','FaceColor','none')
    hold on
    surf(Nuc_n2_X(:,:,k),Nuc_n2_Y(:,:,k),Nuc_n2_Z(:,:,k),'EdgeColor','b','FaceColor','none')
    title(name_str)
    xlabel('X (\mum)')
    ylabel('Y (\mum)')
    zlabel('Z (\mum)')
    set(gca,'DataAspectRatio',[1 1 1])
    hold off
    grid off
    view(90,0)
    
    subplot(1,3,3)
    name_str = strcat('SM',num2str(k));
    surf(Nuc_p2_X(:,:,k),Nuc_p2_Y(:,:,k),Nuc_p2_Z(:,:,k),'EdgeColor','r','FaceColor','none')
    hold on
    surf(Nuc_n2_X(:,:,k),Nuc_n2_Y(:,:,k),Nuc_n2_Z(:,:,k),'EdgeColor','b','FaceColor','none')
    title(name_str)
    xlabel('X (\mum)')
    ylabel('Y (\mum)')
    zlabel('Z (\mum)')
    set(gca,'DataAspectRatio',[1 1 1])
    hold off
    grid off
    view(0,90)

end

%% SH Functions

function [SHE_Cell_0_Coef_C,SHE_Cell_0_Coef_S] = SHE_Formatter(SHE_Cell_0)

    SHE_Cell_0_Coef_C = zeros(17,17);
    index = 1;
    for l = 0:16
        SHE_Cell_0_Coef_C(l+1,1:(l+1)) = SHE_Cell_0(index:(index+l));
        index = index + l + 1;
    end

    SHE_Cell_0_Coef_S = zeros(17,17);
    index = 154;
    for l = 1:16
        SHE_Cell_0_Coef_S(l+1,2:(l+1)) = SHE_Cell_0(index:(index+l-1));
        index = index + l;
    end
end

function [X,Y,Z] = SHE_2_Cart(SHE_Coef_Cell_C,SHE_Coef_Cell_S,Resolution)

    % Create the unit sphere mesh

    az = linspace(-pi,pi,Resolution);
    el = linspace(0,pi,Resolution);
    [phi,theta] = meshgrid(az,el);

    C_Total = zeros(Resolution,Resolution);
    S_Total = zeros(Resolution,Resolution);
    for l = 0:16
        for m = 0:l

            Plm = legendre(l,cos(theta));
            if l ~= 0
                Plm = reshape(Plm(m+1,:,:),size(phi));
            end
            mneg = -m;
            Plmneg = ((-1)^m).*(factorial(l-m)/factorial(l+m)).*Plm;
            a = (2*l+1)*factorial(l-m);
            b = 4*pi*factorial(l+m);
            N = sqrt(a/b);
            Ylm = N.*Plm.*exp(1i*m*phi);
            Rlm = sqrt(4*pi/((2*l)+1)).*Ylm;
            aneg = (2*l+1)*factorial(l-mneg);
            bneg = 4*pi*factorial(l+mneg);
            Nneg = sqrt(aneg/bneg);
            Ylmneg = Nneg.*Plmneg.*exp(1i*mneg*phi);
            Rlm = sqrt(4*pi/((2*l)+1)).*Ylm;
            Rlmneg = sqrt(4*pi/((2*l)+1)).*Ylmneg;

            if m == 0
                
                C = Rlm;
                S = zeros(size(C));
                if l > 0
                    S = -C;
                end

            else

                C = (1/sqrt(2)).*(((-1)^m).*real(Rlm) + real(Rlmneg));
                S = -C;

            end

            C_Total = C_Total + SHE_Coef_Cell_C(l+1,m+1).*C;
            S_Total = S_Total + SHE_Coef_Cell_S(l+1,m+1).*S;

        end
        fprintf('Completed l = %d \n',l)
    end

    R_Total = C_Total + S_Total;
    [X,Y,Z] = sph2cart(phi, pi/2-theta, R_Total);

end
