%% Import SHE Coefficient Data

clear all
clc
close all force

M = readtable("metadata.csv");
Important_Columns = [49:337,627:915,338:626,916:1204];

M_Important = M{:,Important_Columns};

%% Extract SHE Coefficients in 17x17

ROI = 2001;
SHE_Coef = M_Important(ROI,:);

SHE_Coef_Nuc_C = zeros(17,17);
SHE_Coef_Cell_C = zeros(17,17);
SHE_Coef_Nuc_S = zeros(17,17);
SHE_Coef_Cell_S = zeros(17,17);
for l = 0:16
    SHE_Coef_Nuc_C(l+1,:) = SHE_Coef((17*l+1):(17*(l+1)));
    SHE_Coef_Cell_C(l+1,:) = SHE_Coef((17*l+290):(17*(l+1)+289));
    SHE_Coef_Nuc_S(l+1,:) = SHE_Coef((17*l+579):(17*(l+1)+578));
    SHE_Coef_Cell_S(l+1,:) = SHE_Coef((17*l+868):(17*(l+1)+867));
end

%% Plotting 3D Images

[X,Y,Z] = SHE_2_Cart(SHE_Coef_Cell_C,SHE_Coef_Cell_S,100);

figure
surf(X,Y,Z,'EdgeColor','m','FaceColor','none')
title('Cell Shape')
xlabel('X')
ylabel('Y')
zlabel('Z')

%% SHE to Spherical Coordinates Functions

function [X,Y,Z] = SHE_2_Cart(SHE_Coef_Cell_C,SHE_Coef_Cell_S,Resolution)

    dx = pi/Resolution;
    col = 0:dx:pi;
    az = 0:dx:2*pi;
    [phi,theta] = meshgrid(az,col);

    Ylm_All = zeros(Resolution+1,2*Resolution+1);
    for l = 0:16
        for m = 0:l

            Plm = legendre(l,cos(theta));
            if l ~= 0
                Plm = reshape(Plm(m+1,:,:),size(phi));
            end
            
            a = (2*l+1)*factorial(l-m);
            b = 4*pi*factorial(l+m);
            C = sqrt(a/b);

            Coef = SHE_Coef_Cell_C(l+1,m+1);

            Ylm_All = Ylm_All + Coef.*C.*Plm.*exp(1i*m*phi);

        end
    end

    for l = 0:16
        for m = 0:l

            Plm = legendre(l,cos(theta));
            if l ~= 0
                Plm = reshape(Plm(m+1,:,:),size(phi));
            end
            
            a = (2*l+1)*factorial(l-m);
            b = 4*pi*factorial(l+m);
            C = sqrt(a/b);

            Coef = SHE_Coef_Cell_S(l+1,m+1);

            Ylm_All = Ylm_All + Coef.*C.*Plm.*exp(1i*m*phi);

        end
    end

    [X,Y,Z] = sph2cart(phi, pi/2-theta, abs(real(Ylm_All)));

end
