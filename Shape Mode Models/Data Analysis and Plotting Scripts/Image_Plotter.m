%% Loading Image Data

clear all
close all
clc

IM1 = BioformatsImage('C:\Users\sebas\OneDrive\Desktop\Actin_1\AICS-16_5267_413524.ome.tif');

x = IM1.width;
y = IM1.height;
z = IM1.sizeZ;

IM1_ACT = zeros(y,x,z,'uint16');
IM1_MEM = zeros(y,x,z,'uint16');
IM1_MEM_FILL = zeros(y,x,z,'uint16');
IM1_NUC = zeros(y,x,z,'uint16');

for k = 1:z
    IM = getPlane(IM1,k,5,1);
    IM1_ACT(:,:,k) = IM;
    IM = getPlane(IM1,k,6,1);
    IM1_MEM_FILL(:,:,k) = IM;
    IM = getPlane(IM1,k,8,1);
    IM1_MEM(:,:,k) = IM;
    IM = getPlane(IM1,k,9,1);
    IM1_NUC(:,:,k) = IM;
end

IM1_ACT_BOUND = IM1_ACT;

for kz = 1:z
    for ky = 1:y
        for kx = 1:x
            if IM1_MEM_FILL(ky,kx,kz) == 0
                IM1_ACT_BOUND(ky,kx,kz) = 0;
            end
        end
    end
end

%% Mapping Image Data to 3D Coordinates

Max_x = 495;
Max_y = 476;
Max_z = 68;

IM1_ACT_BOUND_Standard = zeros(Max_y,Max_x,Max_z,'int8');
IM1_ACT_BOUND_Standard(1:y,1:x,1:z) = int8(mod(IM1_ACT_BOUND,2));

IM1_MEM_Standard = zeros(Max_y,Max_x,Max_z,'int8');
IM1_MEM_Standard(1:y,1:x,1:z) = int8(mod(IM1_MEM,2));

IM1_NUC_Standard = zeros(Max_y,Max_x,Max_z,'int8');
IM1_NUC_Standard(1:y,1:x,1:z) = int8(mod(IM1_NUC,2));

ACT_BOUND_X = [];
ACT_BOUND_Y = [];
ACT_BOUND_Z = [];

MEM_X = [];
MEM_Y = [];
MEM_Z = [];

NUC_X = [];
NUC_Y = [];
NUC_Z = [];

for kz = 1:z
    for ky = 1:y
        for kx = 1:x
            if IM1_ACT_BOUND_Standard(ky,kx,kz) == 1
                ACT_BOUND_X(length(ACT_BOUND_X)+1) = kx;
                ACT_BOUND_Y(length(ACT_BOUND_Y)+1) = ky;
                ACT_BOUND_Z(length(ACT_BOUND_Z)+1) = kz;
            end
            if IM1_MEM_Standard(ky,kx,kz) == 1
                MEM_X(length(MEM_X)+1) = kx;
                MEM_Y(length(MEM_Y)+1) = ky;
                MEM_Z(length(MEM_Z)+1) = kz;
            end
            if IM1_NUC_Standard(ky,kx,kz) == 1
                NUC_X(length(NUC_X)+1) = kx;
                NUC_Y(length(NUC_Y)+1) = ky;
                NUC_Z(length(NUC_Z)+1) = kz;
            end
        end
    end
end

%% Plotting Image Data

ACT_BOUND_X = 0.108333*ACT_BOUND_X;
ACT_BOUND_Y = 0.108333*ACT_BOUND_Y;
ACT_BOUND_Z = 0.108333*ACT_BOUND_Z;

MEM_X = 0.108333*MEM_X;
MEM_Y = 0.108333*MEM_Y;
MEM_Z = 0.108333*MEM_Z;

NUC_X = 0.108333*NUC_X;
NUC_Y = 0.108333*NUC_Y;
NUC_Z = 0.108333*NUC_Z;

figure
plot3(MEM_X,MEM_Y,MEM_Z,'.b','MarkerSize',20)
hold on
plot3(NUC_X,NUC_Y,NUC_Z,'.m','MarkerSize',20)
plot3(ACT_BOUND_X,ACT_BOUND_Y,ACT_BOUND_Z,'.k','MarkerSize',20)
title('3D Cell Membrane, Nucleus, and Actin Filaments')
xlabel('X (\mum)')
ylabel('Y (\mum)')
zlabel('Z (\mum)')
