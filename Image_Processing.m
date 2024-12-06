%% Bounding Each of the Actin Images

clear all
close all
clc

Actin_1 = dir('C:\Users\sebas\OneDrive\Desktop\Actin_1');
Actin_Images = struct();
Actin_Images_Nucleus = struct();

for k = 3:302

    Cell_ID = Actin_1(k).name;
    File_Path = strcat('C:\Users\sebas\OneDrive\Desktop\Actin_1\',Cell_ID);
    IM1 = BioformatsImage(File_Path);

    x = IM1.width;
    y = IM1.height;
    z = IM1.sizeZ;
    
    IM1_ACT = zeros(y,x,z,'uint16');
    IM1_MEM = zeros(y,x,z,'uint16');
    IM1_MEM_FILL = zeros(y,x,z,'uint16');
    
    for m = 1:z
        IM = getPlane(IM1,m,5,1);
        IM1_ACT(:,:,m) = IM;
        IM = getPlane(IM1,m,6,1);
        IM1_MEM_FILL(:,:,m) = IM;
        IM = getPlane(IM1,m,8,1);
        IM1_MEM(:,:,m) = IM;
    end
    
    IM1_NUC = zeros(y,x,z,'uint16');
    
    for m = 1:z
        IM = getPlane(IM1,m,9,1);
        IM1_NUC(:,:,m) = IM;
    end
    
    IM1_ACT_BOUND = IM1_ACT;
    
    for kz = 1:z
        for ky = 1:y
            for kx = 1:x
                if IM1_MEM_FILL(ky,kx,kz) == 0
                    IM1_ACT_BOUND(ky,kx,kz) = 0;
                end
                if IM1_MEM(ky,kx,kz) == 255
                    IM1_ACT_BOUND(ky,kx,kz) = 255;
                end
            end
        end
    end
    
    IM1_Byte = int8(mod(IM1_ACT_BOUND,2));
    IM1_Byte2 = int8(mod(IM1_NUC,2));
    
    ImageNum = strcat('Image',num2str(k-2));
    Actin_Images.(ImageNum) = IM1_Byte;
    Actin_Images_Nucleus.(ImageNum) = IM1_Byte2;
    
    Prnt_Msg = strcat(ImageNum,' is Done! \n');
    fprintf(Prnt_Msg);
    
end

%% Removing Blank Images

for k = 1:300
    
    Not_Empty_Flag = 0;
    Empty_Z = [];
    
    ImageNum = strcat('Image',num2str(k));
    IM1 = Actin_Images.(ImageNum);
    IM2 = Actin_Images_Nucleus.(ImageNum);
    
    [y,x,z] = size(IM1);
    
    for kz = 1:z
        for ky = 1:y
            for kx = 1:x
                if IM1(ky,kx,kz) ~= 0
                    Not_Empty_Flag = 1;
                end
            end
        end
        if Not_Empty_Flag == 0
            Empty_Z(length(Empty_Z)+1) = kz;
        end
        Not_Empty_Flag = 0;
    end
    
    IM1_Full = IM1;
    IM1_Full(:,:,Empty_Z) = [];
    
    Actin_Images.(ImageNum) = IM1_Full;
    
    IM2(:,:,Empty_Z) = [];
    Actin_Images_Nucleus.(ImageNum) = IM2;
    
    Prnt_Msg = strcat(ImageNum,' is Done! \n');
    fprintf(Prnt_Msg);
    
end

%% Finding All Image Dimensions

X = [];
Y = [];
Z = [];

for k = 1:300
    
    ImageNum = strcat('Image',num2str(k));
    IM1 = Actin_Images.(ImageNum);
    
    [y,x,z] = size(IM1);
    
    X(length(X)+1) = x;
    Y(length(Y)+1) = y;
    Z(length(Z)+1) = z;
    
end

X = X';
Y = Y';
Z = Z';

%% Saving the Images with Variable Size and Cell ID Numbers

ID_Nums = [];

for k = 1:300
    
    Cell_ID = Actin_1(k+2).name;
    Cell_ID_Num = str2num(Cell_ID((length(Cell_ID)-13):(length(Cell_ID)-8)));
    ID_Nums(k) = Cell_ID_Num;
    
end

Actin_Images.ID = ID_Nums';
Actin_Images_Nucleus.ID = ID_Nums';

m = matfile('Actin_Images_Variable_Size','Writable',true);
m.Actin_Images = Actin_Images;

m = matfile('Actin_Images_Nucleus_Variable_Size','Writable',true);
m.Actin_Images_Nucleus = Actin_Images_Nucleus;

%% Nucleus Images to Standard Size

clear all
close all
clc

load('Actin_Images_Nucleus_Variable_Size')

Max_x = 382;
Max_y = 461;
Max_z = 57;

Actin_Images_Nucleus_Standard_Size_1 = struct();

for k = 1:300
    
    ImageNum = strcat('Image',num2str(k));
    IM1 = Actin_Images_Nucleus.(ImageNum);
    
    [y,x,z] = size(IM1);
    
    IM2 = zeros(Max_y,Max_x,Max_z,'int8');
    IM2(1:y,1:x,1:z) = IM1;
    
    Actin_Images_Nucleus_Standard_Size_1.(ImageNum) = IM2;
    
    Prnt_Msg = strcat(ImageNum,' is Done! \n');
    fprintf(Prnt_Msg);
    
end

Actin_Images_Nucleus_Standard_Size_1.ID = Actin_Images_Nucleus.ID(1:300);

m = matfile('Actin_Images_Nucleus_Standard_Size_1','Writable',true);
m.Actin_Images_Nucleus_Standard_Size_1 = Actin_Images_Nucleus_Standard_Size_1;

%% Scaling Images to Standard Size

clear all
close all
clc

load('Actin_Images_Variable_Size')

Max_x = 495;
Max_y = 476;
Max_z = 68;

Actin_Images_Standard_Size_1 = struct();

for k = 1:300
    
    ImageNum = strcat('Image',num2str(k));
    IM1 = Actin_Images.(ImageNum);
    
    [y,x,z] = size(IM1);
    
    IM2 = zeros(Max_y,Max_x,Max_z,'int8');
    IM2(1:y,1:x,1:z) = IM1;
    
    Actin_Images_Standard_Size_1.(ImageNum) = IM2;
    
    Prnt_Msg = strcat(ImageNum,' is Done! \n');
    fprintf(Prnt_Msg);
    
end

Actin_Images_Standard_Size_1.ID = Actin_Images.ID(1:300);

m = matfile('Actin_Images_Standard_Size_1','Writable',true);
m.Actin_Images_Standard_Size_1 = Actin_Images_Standard_Size_1;
