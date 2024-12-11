%% Loading In Data

clear all
close all
clc

load('SeparatedData.mat')
load('Nuc_ALL_Interphase_Cells_ALL_Includes_Previous_NaN.mat')
load('MLRModels_INTERPHASE_ALL_Z_Scaled_VER_2.mat')
load('PCAMatrices.mat')
load('SMStats.mat')
load('ImNum2CellID.csv')
load('ImNum2CellID_Interphase.csv')

Actin_1 = dir('D:\Q-BIO Research\Actin_Images');

%% Searching for Cells with Small Height

low_H_index = [];

for k = 1:4010
    if Actin(k,5) < 5 && Actin(k,5) > 4
        if Actin(k,2) ~= 1
            H_Cell_ID = Actin(k,3);
            for m = 1:3824
                if ImNum2CellID_Interphase(m,2) == H_Cell_ID
                    low_H_index(length(low_H_index)+1) = m;
                end
            end
        end
    end
end

%% Choose a Cell, Calculate the Points, and Plot

tic
close all

ROI = 1;  % Row of Interest (Up to 3,824)
ToggleX = 0; % If 1, inverts the x axis of the ground truth image

% Retrieving and Calculating Plottable Points

    % NN

Cell_ID_Num = ImNum2CellID_Interphase(ROI,2);

for k = 1:197682
    if Interphase(k,3) == Cell_ID_Num
        Inter_Num = k;
        break
    end
end

SM1_pred_NN = 0;
SM2_pred_NN = 0;
SM3_pred_NN = 0;
SM4_pred_NN = 0;
SM5_pred_NN = 0;
SM6_pred_NN = 0;
SM7_pred_NN = 0;
SM8_pred_NN = 0;
SM9_pred_NN = 0;
SM10_pred_NN = 0;
SM11_pred_NN = 0;
SM12_pred_NN = 0;

for k = 1:10

    net1 = models{k};
    SM_pred = net1(Interphase(Inter_Num,[628:630,596:615])');
    SM1_pred_NN = SM1_pred_NN + 0.1*SM_pred(4,:)';
    SM2_pred_NN = SM2_pred_NN + 0.1*SM_pred(5,:)';
    SM3_pred_NN = SM3_pred_NN + 0.1*SM_pred(6,:)';
    SM4_pred_NN = SM4_pred_NN + 0.1*SM_pred(7,:)';
    SM5_pred_NN = SM5_pred_NN + 0.1*SM_pred(8,:)';
    SM6_pred_NN = SM6_pred_NN + 0.1*SM_pred(9,:)';
    SM7_pred_NN = SM7_pred_NN + 0.1*SM_pred(10,:)';
    SM8_pred_NN = SM8_pred_NN + 0.1*SM_pred(11,:)';
    SM9_pred_NN = SM9_pred_NN + 0.1*SM_pred(12,:)';
    SM10_pred_NN = SM10_pred_NN + 0.1*SM_pred(13,:)';
    SM11_pred_NN = SM11_pred_NN + 0.1*SM_pred(14,:)';
    SM12_pred_NN = SM12_pred_NN + 0.1*SM_pred(15,:)';

end

SM1_pred_NN = SM1_pred_NN.*SM_nuc_Inter_std(1) + SM_nuc_Inter_mean(1);
SM2_pred_NN = SM2_pred_NN.*SM_nuc_Inter_std(2) + SM_nuc_Inter_mean(2);
SM3_pred_NN = SM3_pred_NN.*SM_nuc_Inter_std(3) + SM_nuc_Inter_mean(3);
SM4_pred_NN = SM4_pred_NN.*SM_nuc_Inter_std(4) + SM_nuc_Inter_mean(4);
SM5_pred_NN = SM5_pred_NN.*SM_nuc_Inter_std(5) + SM_nuc_Inter_mean(5);
SM6_pred_NN = SM6_pred_NN.*SM_nuc_Inter_std(6) + SM_nuc_Inter_mean(6);
SM7_pred_NN = SM7_pred_NN.*SM_nuc_Inter_std(7) + SM_nuc_Inter_mean(7);
SM8_pred_NN = SM8_pred_NN.*SM_nuc_Inter_std(8) + SM_nuc_Inter_mean(8);
SM9_pred_NN = SM9_pred_NN.*SM_nuc_Inter_std(9) + SM_nuc_Inter_mean(9);
SM10_pred_NN = SM10_pred_NN.*SM_nuc_Inter_std(10) + SM_nuc_Inter_mean(10);
SM11_pred_NN = SM11_pred_NN.*SM_nuc_Inter_std(11) + SM_nuc_Inter_mean(11);
SM12_pred_NN = SM12_pred_NN.*SM_nuc_Inter_std(12) + SM_nuc_Inter_mean(12);

SM_pred_NN = [SM1_pred_NN,SM2_pred_NN,SM3_pred_NN,SM4_pred_NN,SM5_pred_NN,SM6_pred_NN,SM7_pred_NN,SM8_pred_NN,SM9_pred_NN,SM10_pred_NN,SM11_pred_NN,SM12_pred_NN];

SHE_pred_NN = SM_pred_NN*PCA_nuc_Inter(:,1:12)';

SHE_Coef_C_NN = zeros(17,17);
index = 1;
for l = 0:16
    SHE_Coef_C_NN(l+1,1:(l+1)) = SHE_pred_NN(index:(index+l));
    index = index + l + 1;
end

SHE_Coef_S_NN = zeros(17,17);
index = 154;
for l = 1:16
    SHE_Coef_S_NN(l+1,2:(l+1)) = SHE_pred_NN(index:(index+l-1));
    index = index + l;
end

[NN_X,NN_Y,NN_Z] = SHE_2_Cart(SHE_Coef_C_NN,SHE_Coef_S_NN,100);

NN_X = 0.108333.*NN_X;
NN_Y = 0.108333.*NN_Y;
NN_Z = 0.108333.*NN_Z;

NN_X = NN_X - min(min(NN_X));
NN_Y = NN_Y - min(min(NN_Y));
NN_Z = NN_Z - min(min(NN_Z));

    % MLR

SM1_pred_MLR = SM1.Fitted(Inter_Num);
SM2_pred_MLR = SM2.Fitted(Inter_Num);
SM3_pred_MLR = SM3.Fitted(Inter_Num);
SM4_pred_MLR = SM4.Fitted(Inter_Num);
SM5_pred_MLR = SM5.Fitted(Inter_Num);
SM6_pred_MLR = SM6.Fitted(Inter_Num);
SM7_pred_MLR = SM7.Fitted(Inter_Num);
SM8_pred_MLR = SM8.Fitted(Inter_Num);
SM9_pred_MLR = SM9.Fitted(Inter_Num);
SM10_pred_MLR = SM10.Fitted(Inter_Num);
SM11_pred_MLR = SM11.Fitted(Inter_Num);
SM12_pred_MLR = SM12.Fitted(Inter_Num);

SM1_pred_MLR = SM1_pred_MLR.*SM_nuc_Inter_std(1) + SM_nuc_Inter_mean(1);
SM2_pred_MLR = SM2_pred_MLR.*SM_nuc_Inter_std(2) + SM_nuc_Inter_mean(2);
SM3_pred_MLR = SM3_pred_MLR.*SM_nuc_Inter_std(3) + SM_nuc_Inter_mean(3);
SM4_pred_MLR = SM4_pred_MLR.*SM_nuc_Inter_std(4) + SM_nuc_Inter_mean(4);
SM5_pred_MLR = SM5_pred_MLR.*SM_nuc_Inter_std(5) + SM_nuc_Inter_mean(5);
SM6_pred_MLR = SM6_pred_MLR.*SM_nuc_Inter_std(6) + SM_nuc_Inter_mean(6);
SM7_pred_MLR = SM7_pred_MLR.*SM_nuc_Inter_std(7) + SM_nuc_Inter_mean(7);
SM8_pred_MLR = SM8_pred_MLR.*SM_nuc_Inter_std(8) + SM_nuc_Inter_mean(8);
SM9_pred_MLR = SM9_pred_MLR.*SM_nuc_Inter_std(9) + SM_nuc_Inter_mean(9);
SM10_pred_MLR = SM10_pred_MLR.*SM_nuc_Inter_std(10) + SM_nuc_Inter_mean(10);
SM11_pred_MLR = SM11_pred_MLR.*SM_nuc_Inter_std(11) + SM_nuc_Inter_mean(11);
SM12_pred_MLR = SM12_pred_MLR.*SM_nuc_Inter_std(12) + SM_nuc_Inter_mean(12);

SM_pred_MLR = [SM1_pred_MLR,SM2_pred_MLR,SM3_pred_MLR,SM4_pred_MLR,SM5_pred_MLR,SM6_pred_MLR,SM7_pred_MLR,SM8_pred_MLR,SM9_pred_MLR,SM10_pred_MLR,SM11_pred_MLR,SM12_pred_MLR];

SHE_pred_MLR = SM_pred_MLR*PCA_nuc_Inter(:,1:12)';

SHE_Coef_C_MLR = zeros(17,17);
index = 1;
for l = 0:16
    SHE_Coef_C_MLR(l+1,1:(l+1)) = SHE_pred_MLR(index:(index+l));
    index = index + l + 1;
end

SHE_Coef_S_MLR = zeros(17,17);
index = 154;
for l = 1:16
    SHE_Coef_S_MLR(l+1,2:(l+1)) = SHE_pred_MLR(index:(index+l-1));
    index = index + l;
end

[MLR_X,MLR_Y,MLR_Z] = SHE_2_Cart(SHE_Coef_C_MLR,SHE_Coef_S_MLR,100);

MLR_X = 0.108333.*MLR_X;
MLR_Y = 0.108333.*MLR_Y;
MLR_Z = 0.108333.*MLR_Z;

MLR_X = MLR_X - min(min(MLR_X));
MLR_Y = MLR_Y - min(min(MLR_Y));
MLR_Z = MLR_Z - min(min(MLR_Z));

    % SHE

SHE_Coef = Interphase(Inter_Num,307:595);

SHE_Coef_C = zeros(17,17);
index = 1;
for l = 0:16
    SHE_Coef_C(l+1,1:(l+1)) = SHE_Coef(index:(index+l));
    index = index + l + 1;
end

SHE_Coef_S = zeros(17,17);
index = 154;
for l = 1:16
    SHE_Coef_S(l+1,2:(l+1)) = SHE_Coef(index:(index+l-1));
    index = index + l;
end

[SHE_X,SHE_Y,SHE_Z] = SHE_2_Cart(SHE_Coef_C,SHE_Coef_S,100);

SHE_X = 0.108333.*SHE_X;
SHE_Y = 0.108333.*SHE_Y;
SHE_Z = 0.108333.*SHE_Z;

SHE_X = SHE_X - min(min(SHE_X));
SHE_Y = SHE_Y - min(min(SHE_Y));
SHE_Z = SHE_Z - min(min(SHE_Z));

    % True Nucleus Image

for k = 1:4010
    if ImNum2CellID(k,2) == Cell_ID_Num
        Dir_Num = k + 2;
    end
end

Actin_1 = dir('D:\Q-BIO Research\Actin_Images');
File_Name = Actin_1(Dir_Num).name;
File_Path = strcat('D:\Q-BIO Research\Actin_Images\',File_Name);

IM1 = BioformatsImage(File_Path);

x = IM1.width;
y = IM1.height;
z = IM1.sizeZ;

IM1_NUC = zeros(y,x,z,'uint16');

for k = 1:z
    IM = getPlane(IM1,k,9,1);
    IM1_NUC(:,:,k) = IM;
end

Max_x = 495;
Max_y = 476;
Max_z = 75;

IM1_NUC_Standard = zeros(Max_y,Max_x,Max_z,'int8');
IM1_NUC_Standard(1:y,1:x,1:z) = int8(mod(IM1_NUC,2));

NUC_X = [];
NUC_Y = [];
NUC_Z = [];

for kz = 1:z
    for ky = 1:y
        for kx = 1:x
            if IM1_NUC_Standard(ky,kx,kz) == 1
                NUC_X(length(NUC_X)+1) = kx;
                NUC_Y(length(NUC_Y)+1) = ky;
                NUC_Z(length(NUC_Z)+1) = kz;
            end
        end
    end
end

NUC_X = 0.108333*NUC_X;
NUC_Y = 0.108333*NUC_Y;
NUC_Z = 0.29*NUC_Z;

    % True Cell Image

IM1_CELL = zeros(y,x,z,'uint16');

for k = 1:z
    IM = getPlane(IM1,k,8,1);
    IM1_CELL(:,:,k) = IM;
end

Max_x = 495;
Max_y = 476;
Max_z = 75;

IM1_CELL_Standard = zeros(Max_y,Max_x,Max_z,'int8');
IM1_CELL_Standard(1:y,1:x,1:z) = int8(mod(IM1_CELL,2));

Num_points = 0;
Num_points_max = 0;

for kz = 1:z
    for ky = 1:y
        for kx = 1:x
            if IM1_CELL_Standard(ky,kx,kz) == 1
                Num_points = Num_points + 1;
            end
        end
    end
    if Num_points > Num_points_max
        Num_points_max = Num_points;
    end
    Num_points = 0;
end

CELL_XYZ = zeros(Num_points_max,3,z);

for kz = 1:z
    counter = 1;
    for ky = 1:y
        for kx = 1:x
            if IM1_CELL_Standard(ky,kx,kz) == 1
                CELL_XYZ(counter,:,kz) = [0.108333*kx,0.108333*ky,0.29*kz];
                counter = counter + 1;
            end
        end
    end
end

% p = [NUC_X',NUC_Y',NUC_Z'];
% [t] = MyCrustOpen(p);

% Fixing Rotation

max_dist_CELL = 0;
max_xy_pair = [];
max_z_pair = [];

for kz = 1:z
    counter = 2;
    for kxy = 1:Num_points_max
        for kxy2 = counter:Num_points_max
            
            xy_pair = [CELL_XYZ(kxy,1:2,kz),CELL_XYZ(kxy2,1:2,kz)];
            
            if any(xy_pair(1:2)) && any(xy_pair(3:4))

                dist_pair = sqrt((xy_pair(1) - xy_pair(3))^2 + (xy_pair(2) - xy_pair(4))^2);
                
                if dist_pair > max_dist_CELL
                    max_dist_CELL = dist_pair;
                    max_xy_pair = xy_pair;
                    max_z_pair = kz;
                end
            end
        end
        counter = counter + 1;
    end
end

xy_vector = max_xy_pair(3:4) - max_xy_pair(1:2);
x_axis = [1,0];

cell_angle = pi - acos(dot(xy_vector,x_axis)/max_dist_CELL);

% figure
% plot3(CELL_XYZ(1:659,1,10),CELL_XYZ(1:659,2,10),CELL_XYZ(1:659,3,10))
% title('Cell Unrotated')
% xlabel('X (\mum)')
% ylabel('Y (\mum)')
% zlabel('Z (\mum)')
% ax = gca; 
% ax.FontSize = 36;
% view(0,90)
% 
% CELL_XYZ_unpaired = CELL_XYZ(1:659,:,10);
% 
rot_mat = [cos(cell_angle),-sin(cell_angle),0;sin(cell_angle),cos(cell_angle),0;0,0,1];
% CELL_XYZ_rot = rot_mat*CELL_XYZ_unpaired';
%
% figure
% plot3(CELL_XYZ_rot(1,:),CELL_XYZ_rot(2,:),CELL_XYZ_rot(3,:))
% title('Cell Rotated')
% xlabel('X (\mum)')
% ylabel('Y (\mum)')
% zlabel('Z (\mum)')
% ax = gca; 
% ax.FontSize = 36;
% view(0,90)

NUC_XYZ = [NUC_X',NUC_Y',NUC_Z'];
NUC_XYZ_rot = rot_mat*NUC_XYZ';

NUC_X_rot = NUC_XYZ_rot(1,:);
NUC_Y_rot = NUC_XYZ_rot(2,:);
NUC_Z_rot = NUC_XYZ_rot(3,:);

NUC_X_rot = NUC_X_rot - min(NUC_X_rot);
NUC_Y_rot = NUC_Y_rot - min(NUC_Y_rot);
NUC_Z_rot = NUC_Z_rot - min(NUC_Z_rot);

figure
plot3(NUC_X,NUC_Y,NUC_Z,'.m')
title('Ground-Truth Nucleus Image Unrotated')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
grid on
ax = gca;
ax.FontSize = 36;

figure
plot3(NUC_X_rot,NUC_Y_rot,NUC_Z_rot,'.m')
title('Ground-Truth Nucleus Image Rotated')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
grid on
ax = gca;
ax.FontSize = 36;

% Toggling X axis

if ToggleX == 1
    NUC_X_rot = max(NUC_X_rot) - NUC_X_rot;
end

% Plotting All Figures

    % NN

NN_lims = [max(max(NN_X)),max(max(NN_Y)),max(max(NN_Z))];
MLR_lims = [max(max(MLR_X)),max(max(MLR_Y)),max(max(MLR_Z))];
SHE_lims = [max(max(SHE_X)),max(max(SHE_Y)),max(max(SHE_Z))];
NUC_lims = [max(NUC_X_rot),max(NUC_Y_rot),max(NUC_Z_rot)];

xyz_lims = [NN_lims;MLR_lims;SHE_lims;NUC_lims];
xyz_lims = max(xyz_lims);

NN_X = 0.5*(xyz_lims(1) - NN_lims(1)) + NN_X;
NN_Y = 0.5*(xyz_lims(2) - NN_lims(2)) + NN_Y;
NN_Z = 0.5*(xyz_lims(3) - NN_lims(3)) + NN_Z;

MLR_X = 0.5*(xyz_lims(1) - MLR_lims(1)) + MLR_X;
MLR_Y = 0.5*(xyz_lims(2) - MLR_lims(2)) + MLR_Y;
MLR_Z = 0.5*(xyz_lims(3) - MLR_lims(3)) + MLR_Z;

SHE_X = 0.5*(xyz_lims(1) - SHE_lims(1)) + SHE_X;
SHE_Y = 0.5*(xyz_lims(2) - SHE_lims(2)) + SHE_Y;
SHE_Z = 0.5*(xyz_lims(3) - SHE_lims(3)) + SHE_Z;

NUC_X_rot = 0.5*(xyz_lims(1) - NUC_lims(1)) + NUC_X_rot;
NUC_Y_rot = 0.5*(xyz_lims(2) - NUC_lims(2)) + NUC_Y_rot;
NUC_Z_rot = 0.5*(xyz_lims(3) - NUC_lims(3)) + NUC_Z_rot;

figure
surf(NN_X,NN_Y,NN_Z,'EdgeColor','m','FaceColor','none')
title('NN Shape Prediction')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca; 
ax.FontSize = 36;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];

    % MLR

figure
surf(MLR_X,MLR_Y,MLR_Z,'EdgeColor','m','FaceColor','none')
title('MLR Shape Prediction')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca; 
ax.FontSize = 36;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];

    % SHE

figure
surf(SHE_X,SHE_Y,SHE_Z,'EdgeColor','m','FaceColor','none')
title('Spherical Harmonic Expansion')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca; 
ax.FontSize = 36;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];

    % True Nucleus Image

figure
plot3(NUC_X_rot,NUC_Y_rot,NUC_Z_rot,'.m')
title('Ground-Truth Nucleus Image')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
grid on
ax = gca;
ax.FontSize = 36;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];

    % Subplot Tiling

hFig = figure('Name',['Cell ID - ',num2str(Cell_ID_Num)]);

subplot(4,3,7)
surf(NN_X,NN_Y,NN_Z,'EdgeColor','m','FaceColor','none')
title('NN Shape Prediction')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];
view(0,0)

subplot(4,3,8)
surf(NN_X,NN_Y,NN_Z,'EdgeColor','m','FaceColor','none')
title('NN Shape Prediction')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];
view(90,0)

subplot(4,3,9)
surf(NN_X,NN_Y,NN_Z,'EdgeColor','m','FaceColor','none')
title('NN Shape Prediction')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];
view(0,90)

subplot(4,3,10)
surf(MLR_X,MLR_Y,MLR_Z,'EdgeColor','m','FaceColor','none')
title('MLR Shape Prediction')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];
view(0,0)

subplot(4,3,11)
surf(MLR_X,MLR_Y,MLR_Z,'EdgeColor','m','FaceColor','none')
title('MLR Shape Prediction')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];
view(90,0)

subplot(4,3,12)
surf(MLR_X,MLR_Y,MLR_Z,'EdgeColor','m','FaceColor','none')
title('MLR Shape Prediction')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];
view(0,90)

subplot(4,3,4)
surf(SHE_X,SHE_Y,SHE_Z,'EdgeColor',[0.4660 0.6740 0.1880],'FaceColor','none')
title('Spherical Harmonic Expansion')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];
view(0,0)

subplot(4,3,5)
surf(SHE_X,SHE_Y,SHE_Z,'EdgeColor',[0.4660 0.6740 0.1880],'FaceColor','none')
title('Spherical Harmonic Expansion')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];
view(90,0)

subplot(4,3,6)
surf(SHE_X,SHE_Y,SHE_Z,'EdgeColor',[0.4660 0.6740 0.1880],'FaceColor','none')
title('Spherical Harmonic Expansion')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
ax = gca;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];
view(0,90)

subplot(4,3,1)
plot3(NUC_X_rot,NUC_Y_rot,NUC_Z_rot,'.b')
title('Ground-Truth Nucleus Image')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
grid on
ax = gca;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];
view(0,0)

subplot(4,3,2)
plot3(NUC_X_rot,NUC_Y_rot,NUC_Z_rot,'.b')
title('Ground-Truth Nucleus Image')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
grid on
ax = gca;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];
view(90,0)

subplot(4,3,3)
plot3(NUC_X_rot,NUC_Y_rot,NUC_Z_rot,'.b')
title('Ground-Truth Nucleus Image')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
grid on
ax = gca;
ax.XLim = [0,xyz_lims(1)];
ax.YLim = [0,xyz_lims(2)];
ax.ZLim = [0,xyz_lims(3)];
view(0,90)

toc

%% Functions

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
