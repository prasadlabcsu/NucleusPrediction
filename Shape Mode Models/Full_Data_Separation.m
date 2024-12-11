%% Dataset Importation

clear all
clc
close all force

M = readtable("metadata.csv");

[M_height, M_width] = size(M);

Important_Columns = [1,13,18,40,41,42,43,44,45,46,1206,1207,1208,1209,1210,1211,1212,1213];
    % 1: Cell ID Numbers
    % 13: Edge Flag (1 = edge cell)
    % 18: Cell Stage (M0 for interphase)
    % 40: Outlier (Yes = outlier)
    % 41: Nuclear Volume (Need to change units)
    % 42: Nuclear Height (Need to change units)
    % 43: Nuclear Area (Need to change units)
    % 44: Membrane Volume (Need to change units)
    % 45: Membrane Height (Need to change units)
    % 46: Membrane Area (Need to change units)
    % 1206-1213: PCA SHE coefficients

M_Important = zeros(215081, 18);

M_Important(:,1) = M{:,1};
M_Important(:,2) = M{:,13};
for k = 1:M_height
    if string(M{k,18}) ~= "M0"
        M_Important(k,3) = 1;
    end
    if string(M{k,40}) == "Yes"
        M_Important(k,4) = 1;
    end
end
M_Important(:,5) = (0.108333^3)*M{:,41};
M_Important(:,6) = (0.108333)*M{:,42};
M_Important(:,7) = (0.108333^2)*M{:,43};
M_Important(:,8) = (0.108333^3)*M{:,44};
M_Important(:,9) = (0.108333)*M{:,45};
M_Important(:,10) = (0.108333^2)*M{:,46};
M_Important(:,11:18) = M{:,1206:1213};

Labeled_Structure = M(:,9);

%% SHE Coefficients

Cell_Columns = 627:1204;
Nuc_Columns = 49:626;

Non_Zero_Columns = [];

M_Cell_SHE = M(:,Cell_Columns);
for k = 1:578
    if table2array(M_Cell_SHE(1:5,k)) ~= zeros(5,1)
        Non_Zero_Columns(length(Non_Zero_Columns) + 1) = k;
    end
end
M_Cell_SHE = table2array(M_Cell_SHE(:,Non_Zero_Columns));

Non_Zero_Columns = [];

M_Nuc_SHE = M(:,Nuc_Columns);
for k = 1:578
    if table2array(M_Nuc_SHE(1:5,k)) ~= zeros(5,1)
        Non_Zero_Columns(length(Non_Zero_Columns) + 1) = k;
    end
end
M_Nuc_SHE = table2array(M_Nuc_SHE(:,Non_Zero_Columns));

% Adding onto the M_Important matrix

M_Important = [M_Important,M_Cell_SHE,M_Nuc_SHE];

% Cell SHE: 19 - 307
% Nuc SHE: 308 - 596

%% Dataset Separation

% Removing outliers

Row_removal_outliers = [];

for k = 1:M_height
    if M_Important(k,4)
        Row_removal_outliers(length(Row_removal_outliers) + 1) = k;
    end
end

M_Important_pared = M_Important;
M_Important_pared(Row_removal_outliers,:) = [];

M_pared = M;
M_pared(Row_removal_outliers,:) = [];

Labeled_Structure_pared = Labeled_Structure;
Labeled_Structure_pared(Row_removal_outliers,:) = [];

% Separating edge cells, non-interphase cells, Actin coefficients

Row_removal_edge = [];
Row_removal_N_edge = [];
Row_removal_inter = [];
Row_removal_N_inter = [];

Row_Actin = [];

for k = 1:length(M_Important_pared)
    if M_Important_pared(k,2) == 1
        Row_removal_edge(length(Row_removal_edge) + 1) = k;
    end
    if M_Important_pared(k,2) == 0
        Row_removal_N_edge(length(Row_removal_N_edge) + 1) = k;
    end
    if M_Important_pared(k,3) == 1
        Row_removal_inter(length(Row_removal_inter) + 1) = k;
    end
    if M_Important_pared(k,3) == 0
        Row_removal_N_inter(length(Row_removal_N_inter) + 1) = k;
    end
    if contains(char(table2array(Labeled_Structure_pared(k,1))),'ACTB')
        Row_Actin(length(Row_Actin) + 1) = k;
    end
end

M_Inter = M_Important_pared;
M_Inter([Row_removal_edge, Row_removal_inter],:) = [];

M_N_Inter = M_Important_pared;
M_N_Inter([Row_removal_edge, Row_removal_N_inter],:) = [];

M_Edge = M_Important_pared;
M_Edge(Row_removal_N_edge,:) = [];

M_Actin = M_Important_pared(Row_Actin,:);

% Interphase and Edge Cell Nuclear and Cell Volume vectors

NV_I = M_Inter(:,4);
NH_I = M_Inter(:,5);
NA_I = M_Inter(:,6);
CV_I = M_Inter(:,7);
CH_I = M_Inter(:,8);
CA_I = M_Inter(:,9);

NV_NI = M_N_Inter(:,4);
NH_NI = M_N_Inter(:,5);
NA_NI = M_N_Inter(:,6);
CV_NI = M_N_Inter(:,7);
CH_NI = M_N_Inter(:,8);
CA_NI = M_N_Inter(:,9);

NV_E = M_Edge(:,4);
NH_E = M_Edge(:,5);
NA_E = M_Edge(:,6);
CV_E = M_Edge(:,7);
CH_E = M_Edge(:,8);
CA_E = M_Edge(:,9);

%% Data Scaling (Interphase and Actin so far)

% PCA Shape Modes (Interphase)

PCA_cell_Inter = pca(M_Inter(:,19:307));
PCA_nuc_Inter = pca(M_Inter(:,308:end));

SM_cell_Inter = [];
SM_nuc_Inter = [];

for k = 1:height(M_Inter)
    for m = 1:20
        SM_cell_Inter(k,m) = sum(PCA_cell_Inter(:,m)'.*M_Inter(k,19:307));
    end
end

for k = 1:height(M_Inter)
    for m = 1:12
        SM_nuc_Inter(k,m) = sum(PCA_nuc_Inter(:,m)'.*M_Inter(k,308:end));
    end
end

% Variance Calculations

VAR_cell_Inter = [];
VAR_nuc_Inter = [];

for k = 1:289
    VAR_cell_Inter(:,k) = var(M_Inter(:,18 + k));
    VAR_nuc_Inter(:,k) = var(M_Inter(:,307 + k));
end

VAR_total_cell_Inter = sum(VAR_cell_Inter);
VAR_total_nuc_Inter = sum(VAR_nuc_Inter);

PER_VAR_cell_Inter = VAR_cell_Inter/VAR_total_cell_Inter;
PER_VAR_nuc_Inter = VAR_nuc_Inter/VAR_total_nuc_Inter;

VAR_SM_cell_Inter = [];
VAR_SM_nuc_Inter = [];

for k = 1:20
    VAR_SM_cell_Inter(:,k) = var(SM_cell_Inter(:,k));
end

for k = 1:12
    VAR_SM_nuc_Inter(:,k) = var(SM_nuc_Inter(:,k));
end

VAR_SM_total_cell_Inter = sum(VAR_SM_cell_Inter);
VAR_SM_total_nuc_Inter = sum(VAR_SM_nuc_Inter);

PER_VAR_SM_cell_Inter = VAR_SM_cell_Inter/VAR_total_cell_Inter;
PER_VAR_SM_nuc_Inter = VAR_SM_nuc_Inter/VAR_total_nuc_Inter;

TOTAL_PER_VAR_SM_cell_Inter = 100*sum(PER_VAR_SM_cell_Inter);
TOTAL_PER_VAR_SM_nuc_Inter = 100*sum(PER_VAR_SM_nuc_Inter);

SM_cell_z_Inter = [];
SM_nuc_z_Inter = [];

for k = 1:20
    SM_cell_z_Inter(:,k) = zscore(SM_cell_Inter(:,k),0);
end

for k = 1:12
    SM_nuc_z_Inter(:,k) = zscore(SM_nuc_Inter(:,k),0);
end

% Adding into M_Inter

M_Inter = [M_Inter,SM_cell_z_Inter,SM_nuc_z_Inter];

% Cell SM: 597 - 616
% Nuc SM: 617 - 628



% PCA Shape Modes (Actin)

PCA_cell_Actin = pca(M_Actin(:,19:307));
PCA_nuc_Actin = pca(M_Actin(:,308:end));

SM_cell_Actin = [];
SM_nuc_Actin = [];

for k = 1:height(M_Actin)
    for m = 1:20
        SM_cell_Actin(k,m) = sum(PCA_cell_Actin(:,m)'.*M_Actin(k,19:307));
    end
end

for k = 1:height(M_Actin)
    for m = 1:12
        SM_nuc_Actin(k,m) = sum(PCA_nuc_Actin(:,m)'.*M_Actin(k,308:end));
    end
end

% Variance Calculations

VAR_cell_Actin = [];
VAR_nuc_Actin = [];

for k = 1:289
    VAR_cell_Actin(:,k) = var(M_Inter(:,18 + k));
    VAR_nuc_Actin(:,k) = var(M_Inter(:,307 + k));
end

VAR_total_cell_Actin = sum(VAR_cell_Actin);
VAR_total_nuc_Actin = sum(VAR_nuc_Actin);

PER_VAR_cell_Actin= VAR_cell_Actin/VAR_total_cell_Actin;
PER_VAR_nuc_Actin = VAR_nuc_Actin/VAR_total_nuc_Actin;

VAR_SM_cell_Actin = [];
VAR_SM_nuc_Actin = [];

for k = 1:20
    VAR_SM_cell_Actin(:,k) = var(SM_cell_Actin(:,k));
end

for k = 1:12
    VAR_SM_nuc_Actin(:,k) = var(SM_nuc_Actin(:,k));
end

VAR_SM_total_cell_Actin = sum(VAR_SM_cell_Actin);
VAR_SM_total_nuc_Actin = sum(VAR_SM_nuc_Actin);

PER_VAR_SM_cell_Actin = VAR_SM_cell_Actin/VAR_total_cell_Actin;
PER_VAR_SM_nuc_Actin = VAR_SM_nuc_Actin/VAR_total_nuc_Actin;

TOTAL_PER_VAR_SM_cell_Actin = 100*sum(PER_VAR_SM_cell_Actin);
TOTAL_PER_VAR_SM_nuc_Actin = 100*sum(PER_VAR_SM_nuc_Actin);

SM_cell_z_Actin = [];
SM_nuc_z_Actin = [];

for k = 1:20
    SM_cell_z_Actin(:,k) = zscore(SM_cell_Actin(:,k),0);
end

for k = 1:12
    SM_nuc_z_Actin(:,k) = zscore(SM_nuc_Actin(:,k),0);
end

% Adding into M_Actin

M_Actin = [M_Actin,SM_cell_z_Actin,SM_nuc_z_Actin];

% Cell SM: 596 - 615
% Nuc SM: 616 - 627



% Scaled H, A, V

H_cell_z_Inter = zscore(M_Inter(:,9),0);
A_cell_z_Inter = zscore(M_Inter(:,10),0);
V_cell_z_Inter = zscore(M_Inter(:,8),0);

H_cell_std_Inter = std(M_Inter(:,9),0);
A_cell_std_Inter = std(M_Inter(:,10),0);
V_cell_std_Inter = std(M_Inter(:,8),0);

H_nuc_z_Inter = zscore(M_Inter(:,6),0);
A_nuc_z_Inter = zscore(M_Inter(:,7),0);
V_nuc_z_Inter = zscore(M_Inter(:,5),0);

H_nuc_std_Inter = std(M_Inter(:,6),0);
A_nuc_std_Inter = std(M_Inter(:,7),0);
V_nuc_std_Inter = std(M_Inter(:,5),0);



H_cell_z_Actin = zscore(M_Actin(:,9),0);
A_cell_z_Actin = zscore(M_Actin(:,10),0);
V_cell_z_Actin = zscore(M_Actin(:,8),0);

H_cell_std_Actin = std(M_Actin(:,9),0);
A_cell_std_Actin = std(M_Actin(:,10),0);
V_cell_std_Actin = std(M_Actin(:,8),0);

H_nuc_z_Actin = zscore(M_Actin(:,6),0);
A_nuc_z_Actin = zscore(M_Actin(:,7),0);
V_nuc_z_Actin = zscore(M_Actin(:,5),0);

H_nuc_std_Actin = std(M_Actin(:,6),0);
A_nuc_std_Actin = std(M_Actin(:,7),0);
V_nuc_std_Actin = std(M_Actin(:,5),0);

% Adding into M_Inter and M_actin

M_Inter = [M_Inter,H_cell_z_Inter,A_cell_z_Inter,V_cell_z_Inter,H_nuc_z_Inter,A_nuc_z_Inter,V_nuc_z_Inter];

M_Actin = [M_Actin,H_cell_z_Actin,A_cell_z_Actin,V_cell_z_Actin,H_nuc_z_Actin,A_nuc_z_Actin,V_nuc_z_Actin];

% Cell H: 629
% Cell A: 630
% Cell V: 631
% Nuc H: 632
% Nuc A: 633
% Nuc V: 634

%% Saving Separated Data

m = matfile('SeparatedData_Adj.mat','Writable',true);

m.Interphase = M_Inter;
m.NonInterphase = M_N_Inter;
m.Edge = M_Edge;
m.Actin = M_Actin;
