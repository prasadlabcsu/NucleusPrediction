%% Loading Separated Data

clear all
close all force
clc

load('SeparatedData.mat')

M_Inter = Interphase;
M_N_Inter = NonInterphase;
M_Edge = Edge;
M_Actin = Actin;

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

%% Interphase MLR (ALL)

mdl_ML_H_SM = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,631),'PEnter',10^(-20));
mdl_ML_A_SM = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,632),'PEnter',10^(-20));
mdl_ML_V_SM = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,633),'PEnter',10^(-20));
mdl_ML_SM1 = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,616),'PEnter',10^(-20));
mdl_ML_SM2 = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,617),'PEnter',10^(-20));
mdl_ML_SM3 = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,618),'PEnter',10^(-20));
mdl_ML_SM4 = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,619),'PEnter',10^(-20));
mdl_ML_SM5 = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,620),'PEnter',10^(-20));
mdl_ML_SM6 = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,621),'PEnter',10^(-20));
mdl_ML_SM7 = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,622),'PEnter',10^(-20));
mdl_ML_SM8 = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,623),'PEnter',10^(-20));
mdl_ML_SM9 = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,624),'PEnter',10^(-20));
mdl_ML_SM10 = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,625),'PEnter',10^(-20));
mdl_ML_SM11 = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,626),'PEnter',10^(-20));
mdl_ML_SM12 = stepwiselm(M_Inter(:,[628:630,596:615]),M_Inter(:,627),'PEnter',10^(-20));

m = matfile('MLRModels_INTERPHASE_ALL_Z_Scaled_Ver_2.mat','Writable',true);

m.Volume = mdl_ML_V_SM;
m.Height = mdl_ML_H_SM;
m.Area = mdl_ML_A_SM;
m.SM1 = mdl_ML_SM1;
m.SM2 = mdl_ML_SM2;
m.SM3 = mdl_ML_SM3;
m.SM4 = mdl_ML_SM4;
m.SM5 = mdl_ML_SM5;
m.SM6 = mdl_ML_SM6;
m.SM7 = mdl_ML_SM7;
m.SM8 = mdl_ML_SM8;
m.SM9 = mdl_ML_SM9;
m.SM10 = mdl_ML_SM10;
m.SM11 = mdl_ML_SM11;
m.SM12 = mdl_ML_SM12;

load('MLRModels_INTERPHASE_ALL_Z_Scaled_Ver_2.mat')

PER_EXP_H = PER_EXP_VAR_W_HAV(Height,M_Inter);
PER_EXP_A = PER_EXP_VAR_W_HAV(Area,M_Inter);
PER_EXP_V = PER_EXP_VAR_W_HAV(Volume,M_Inter);
PER_EXP_SM1 = PER_EXP_VAR_W_HAV(SM1,M_Inter);
PER_EXP_SM2 = PER_EXP_VAR_W_HAV(SM2,M_Inter);
PER_EXP_SM3 = PER_EXP_VAR_W_HAV(SM3,M_Inter);
PER_EXP_SM4 = PER_EXP_VAR_W_HAV(SM4,M_Inter);
PER_EXP_SM5 = PER_EXP_VAR_W_HAV(SM5,M_Inter);
PER_EXP_SM6 = PER_EXP_VAR_W_HAV(SM6,M_Inter);
PER_EXP_SM7 = PER_EXP_VAR_W_HAV(SM7,M_Inter);
PER_EXP_SM8 = PER_EXP_VAR_W_HAV(SM8,M_Inter);
PER_EXP_SM9 = PER_EXP_VAR_W_HAV(SM9,M_Inter);
PER_EXP_SM10 = PER_EXP_VAR_W_HAV(SM10,M_Inter);
PER_EXP_SM11 = PER_EXP_VAR_W_HAV(SM11,M_Inter);
PER_EXP_SM12 = PER_EXP_VAR_W_HAV(SM12,M_Inter);

%% Interphase MLR (No HAV)

mdl_ML_H_SM = stepwiselm(M_Inter(:,596:615),M_Inter(:,631),'PEnter',10^(-20));
mdl_ML_A_SM = stepwiselm(M_Inter(:,596:615),M_Inter(:,632),'PEnter',10^(-20));
mdl_ML_V_SM = stepwiselm(M_Inter(:,596:615),M_Inter(:,633),'PEnter',10^(-20));
mdl_ML_SM1 = stepwiselm(M_Inter(:,596:615),M_Inter(:,616),'PEnter',10^(-20));
mdl_ML_SM2 = stepwiselm(M_Inter(:,596:615),M_Inter(:,617),'PEnter',10^(-20));
mdl_ML_SM3 = stepwiselm(M_Inter(:,596:615),M_Inter(:,618),'PEnter',10^(-20));
mdl_ML_SM4 = stepwiselm(M_Inter(:,596:615),M_Inter(:,619),'PEnter',10^(-20));
mdl_ML_SM5 = stepwiselm(M_Inter(:,596:615),M_Inter(:,620),'PEnter',10^(-20));
mdl_ML_SM6 = stepwiselm(M_Inter(:,596:615),M_Inter(:,621),'PEnter',10^(-20));
mdl_ML_SM7 = stepwiselm(M_Inter(:,596:615),M_Inter(:,622),'PEnter',10^(-20));
mdl_ML_SM8 = stepwiselm(M_Inter(:,596:615),M_Inter(:,623),'PEnter',10^(-20));
mdl_ML_SM9 = stepwiselm(M_Inter(:,596:615),M_Inter(:,624),'PEnter',10^(-20));
mdl_ML_SM10 = stepwiselm(M_Inter(:,596:615),M_Inter(:,625),'PEnter',10^(-20));
mdl_ML_SM11 = stepwiselm(M_Inter(:,596:615),M_Inter(:,626),'PEnter',10^(-20));
mdl_ML_SM12 = stepwiselm(M_Inter(:,596:615),M_Inter(:,627),'PEnter',10^(-20));

m = matfile('MLRModels_INTERPHASE_NO_HAV_Z_Scaled.mat','Writable',true);

m.Volume = mdl_ML_V_SM;
m.Height = mdl_ML_H_SM;
m.Area = mdl_ML_A_SM;
m.SM1 = mdl_ML_SM1;
m.SM2 = mdl_ML_SM2;
m.SM3 = mdl_ML_SM3;
m.SM4 = mdl_ML_SM4;
m.SM5 = mdl_ML_SM5;
m.SM6 = mdl_ML_SM6;
m.SM7 = mdl_ML_SM7;
m.SM8 = mdl_ML_SM8;
m.SM9 = mdl_ML_SM9;
m.SM10 = mdl_ML_SM10;
m.SM11 = mdl_ML_SM11;
m.SM12 = mdl_ML_SM12;

load('MLRModels_INTERPHASE_NO_HAV_Z_Scaled.mat')

PER_EXP_H = PER_EXP_VAR_W_SM(Height,M_Inter);
PER_EXP_A = PER_EXP_VAR_W_SM(Area,M_Inter);
PER_EXP_V = PER_EXP_VAR_W_SM(Volume,M_Inter);
PER_EXP_SM1 = PER_EXP_VAR_W_SM(SM1,M_Inter);
PER_EXP_SM2 = PER_EXP_VAR_W_SM(SM2,M_Inter);
PER_EXP_SM3 = PER_EXP_VAR_W_SM(SM3,M_Inter);
PER_EXP_SM4 = PER_EXP_VAR_W_SM(SM4,M_Inter);
PER_EXP_SM5 = PER_EXP_VAR_W_SM(SM5,M_Inter);
PER_EXP_SM6 = PER_EXP_VAR_W_SM(SM6,M_Inter);
PER_EXP_SM7 = PER_EXP_VAR_W_SM(SM7,M_Inter);
PER_EXP_SM8 = PER_EXP_VAR_W_SM(SM8,M_Inter);
PER_EXP_SM9 = PER_EXP_VAR_W_SM(SM9,M_Inter);
PER_EXP_SM10 = PER_EXP_VAR_W_SM(SM10,M_Inter);
PER_EXP_SM11 = PER_EXP_VAR_W_SM(SM11,M_Inter);
PER_EXP_SM12 = PER_EXP_VAR_W_SM(SM12,M_Inter);

%% Interphase MLR (Only SM 1-5)

mdl_ML_H_SM = stepwiselm(M_Inter(:,596:600),M_Inter(:,631),'PEnter',10^(-20));
mdl_ML_A_SM = stepwiselm(M_Inter(:,596:600),M_Inter(:,632),'PEnter',10^(-20));
mdl_ML_V_SM = stepwiselm(M_Inter(:,596:600),M_Inter(:,633),'PEnter',10^(-20));
mdl_ML_SM1 = stepwiselm(M_Inter(:,596:600),M_Inter(:,616),'PEnter',10^(-20));
mdl_ML_SM2 = stepwiselm(M_Inter(:,596:600),M_Inter(:,617),'PEnter',10^(-20));
mdl_ML_SM3 = stepwiselm(M_Inter(:,596:600),M_Inter(:,618),'PEnter',10^(-20));
mdl_ML_SM4 = stepwiselm(M_Inter(:,596:600),M_Inter(:,619),'PEnter',10^(-20));
mdl_ML_SM5 = stepwiselm(M_Inter(:,596:600),M_Inter(:,620),'PEnter',10^(-20));
mdl_ML_SM6 = stepwiselm(M_Inter(:,596:600),M_Inter(:,621),'PEnter',10^(-20));
mdl_ML_SM7 = stepwiselm(M_Inter(:,596:600),M_Inter(:,622),'PEnter',10^(-20));
mdl_ML_SM8 = stepwiselm(M_Inter(:,596:600),M_Inter(:,623),'PEnter',10^(-20));
mdl_ML_SM9 = stepwiselm(M_Inter(:,596:600),M_Inter(:,624),'PEnter',10^(-20));
mdl_ML_SM10 = stepwiselm(M_Inter(:,596:600),M_Inter(:,625),'PEnter',10^(-20));
mdl_ML_SM11 = stepwiselm(M_Inter(:,596:600),M_Inter(:,626),'PEnter',10^(-20));
mdl_ML_SM12 = stepwiselm(M_Inter(:,596:600),M_Inter(:,627),'PEnter',10^(-20));

m = matfile('MLRModels_INTERPHASE_Minus_Z_Scaled.mat','Writable',true);

m.Volume = mdl_ML_V_SM;
m.Height = mdl_ML_H_SM;
m.Area = mdl_ML_A_SM;
m.SM1 = mdl_ML_SM1;
m.SM2 = mdl_ML_SM2;
m.SM3 = mdl_ML_SM3;
m.SM4 = mdl_ML_SM4;
m.SM5 = mdl_ML_SM5;
m.SM6 = mdl_ML_SM6;
m.SM7 = mdl_ML_SM7;
m.SM8 = mdl_ML_SM8;
m.SM9 = mdl_ML_SM9;
m.SM10 = mdl_ML_SM10;
m.SM11 = mdl_ML_SM11;
m.SM12 = mdl_ML_SM12;

load('MLRModels_INTERPHASE_Minus_Z_Scaled.mat')

PER_EXP_H = PER_EXP_VAR_W_SM(Height,M_Inter);
PER_EXP_A = PER_EXP_VAR_W_SM(Area,M_Inter);
PER_EXP_V = PER_EXP_VAR_W_SM(Volume,M_Inter);
PER_EXP_SM1 = PER_EXP_VAR_W_SM(SM1,M_Inter);
PER_EXP_SM2 = PER_EXP_VAR_W_SM(SM2,M_Inter);
PER_EXP_SM3 = PER_EXP_VAR_W_SM(SM3,M_Inter);
PER_EXP_SM4 = PER_EXP_VAR_W_SM(SM4,M_Inter);
PER_EXP_SM5 = PER_EXP_VAR_W_SM(SM5,M_Inter);
PER_EXP_SM6 = PER_EXP_VAR_W_SM(SM6,M_Inter);
PER_EXP_SM7 = PER_EXP_VAR_W_SM(SM7,M_Inter);
PER_EXP_SM8 = PER_EXP_VAR_W_SM(SM8,M_Inter);
PER_EXP_SM9 = PER_EXP_VAR_W_SM(SM9,M_Inter);
PER_EXP_SM10 = PER_EXP_VAR_W_SM(SM10,M_Inter);
PER_EXP_SM11 = PER_EXP_VAR_W_SM(SM11,M_Inter);
PER_EXP_SM12 = PER_EXP_VAR_W_SM(SM12,M_Inter);

%% ANN (My PCA SM with previous NaN)

load('Nuc_ALL_Interphase_Cells_ALL_Includes_Previous_NaN.mat')

RMSE_H = 0;
RMSE_A = 0;
RMSE_V = 0;
RMSE_SM1 = 0;
RMSE_SM2 = 0;
RMSE_SM3 = 0;
RMSE_SM4 = 0;
RMSE_SM5 = 0;
RMSE_SM6 = 0;
RMSE_SM7 = 0;
RMSE_SM8 = 0;
RMSE_SM9 = 0;
RMSE_SM10 = 0;
RMSE_SM11 = 0;
RMSE_SM12 = 0;


for k = 1:10
    net1 = models{k};
    net1_pred = net1(M_Inter(:,[628:630,596:615])');
    net1_res = [M_Inter(:,[631:633,616:627])] - net1_pred';
    
    RMSE_H = RMSE_H + sqrt(sum((net1_res(:,1)).^2)/height(net1_res(:,1)))*1.6985;
    RMSE_A = RMSE_A + sqrt(sum((net1_res(:,2)).^2)/height(net1_res(:,2)))*107.82;
    RMSE_V = RMSE_V + sqrt(sum((net1_res(:,3)).^2)/height(net1_res(:,3)))*156.5;
    RMSE_SM1 = RMSE_SM1 + sqrt(sum((net1_res(:,4)).^2)/height(net1_res(:,4)));
    RMSE_SM2 = RMSE_SM2 + sqrt(sum((net1_res(:,5)).^2)/height(net1_res(:,5)));
    RMSE_SM3 = RMSE_SM3 + sqrt(sum((net1_res(:,6)).^2)/height(net1_res(:,6)));
    RMSE_SM4 = RMSE_SM4 + sqrt(sum((net1_res(:,7)).^2)/height(net1_res(:,7)));
    RMSE_SM5 = RMSE_SM5 + sqrt(sum((net1_res(:,8)).^2)/height(net1_res(:,8)));
    RMSE_SM6 = RMSE_SM6 + sqrt(sum((net1_res(:,9)).^2)/height(net1_res(:,9)));
    RMSE_SM7 = RMSE_SM7 + sqrt(sum((net1_res(:,10)).^2)/height(net1_res(:,10)));
    RMSE_SM8 = RMSE_SM8 + sqrt(sum((net1_res(:,11)).^2)/height(net1_res(:,11)));
    RMSE_SM9 = RMSE_SM9 + sqrt(sum((net1_res(:,12)).^2)/height(net1_res(:,12)));
    RMSE_SM10 = RMSE_SM10 + sqrt(sum((net1_res(:,13)).^2)/height(net1_res(:,13)));
    RMSE_SM11 = RMSE_SM11 + sqrt(sum((net1_res(:,14)).^2)/height(net1_res(:,14)));
    RMSE_SM12 = RMSE_SM12 + sqrt(sum((net1_res(:,15)).^2)/height(net1_res(:,15)));

end

RMSE_H = RMSE_H/10
RMSE_A = RMSE_A/10
RMSE_V = RMSE_V/10
RMSE_SM1 = RMSE_SM1/10
RMSE_SM2 = RMSE_SM2/10
RMSE_SM3 = RMSE_SM3/10
RMSE_SM4 = RMSE_SM4/10
RMSE_SM5 = RMSE_SM5/10
RMSE_SM6 = RMSE_SM6/10
RMSE_SM7 = RMSE_SM7/10
RMSE_SM8 = RMSE_SM8/10
RMSE_SM9 = RMSE_SM9/10
RMSE_SM10 = RMSE_SM10/10
RMSE_SM11 = RMSE_SM11/10
RMSE_SM12 = RMSE_SM12/10

%% ANN (My PCA SM with previous NaN, No Cell HAV)

load('Nuc_ALL_Interphase_Cells_ALL_SM_WO_HAV_Includes_Previous_NaN.mat')

RMSE_H = 0;
RMSE_A = 0;
RMSE_V = 0;
RMSE_SM1 = 0;
RMSE_SM2 = 0;
RMSE_SM3 = 0;
RMSE_SM4 = 0;
RMSE_SM5 = 0;
RMSE_SM6 = 0;
RMSE_SM7 = 0;
RMSE_SM8 = 0;
RMSE_SM9 = 0;
RMSE_SM10 = 0;
RMSE_SM11 = 0;
RMSE_SM12 = 0;


for k = 1:10
    net1 = models{k};
    net1_pred = net1(M_Inter(:,596:615)');
    net1_res = [M_Inter(:,[631:633,616:627])] - net1_pred';
    
    RMSE_H = RMSE_H + sqrt(sum((net1_res(:,1)).^2)/height(net1_res(:,1)))*1.6985;
    RMSE_A = RMSE_A + sqrt(sum((net1_res(:,2)).^2)/height(net1_res(:,2)))*107.82;
    RMSE_V = RMSE_V + sqrt(sum((net1_res(:,3)).^2)/height(net1_res(:,3)))*156.5;
    RMSE_SM1 = RMSE_SM1 + sqrt(sum((net1_res(:,4)).^2)/height(net1_res(:,4)));
    RMSE_SM2 = RMSE_SM2 + sqrt(sum((net1_res(:,5)).^2)/height(net1_res(:,5)));
    RMSE_SM3 = RMSE_SM3 + sqrt(sum((net1_res(:,6)).^2)/height(net1_res(:,6)));
    RMSE_SM4 = RMSE_SM4 + sqrt(sum((net1_res(:,7)).^2)/height(net1_res(:,7)));
    RMSE_SM5 = RMSE_SM5 + sqrt(sum((net1_res(:,8)).^2)/height(net1_res(:,8)));
    RMSE_SM6 = RMSE_SM6 + sqrt(sum((net1_res(:,9)).^2)/height(net1_res(:,9)));
    RMSE_SM7 = RMSE_SM7 + sqrt(sum((net1_res(:,10)).^2)/height(net1_res(:,10)));
    RMSE_SM8 = RMSE_SM8 + sqrt(sum((net1_res(:,11)).^2)/height(net1_res(:,11)));
    RMSE_SM9 = RMSE_SM9 + sqrt(sum((net1_res(:,12)).^2)/height(net1_res(:,12)));
    RMSE_SM10 = RMSE_SM10 + sqrt(sum((net1_res(:,13)).^2)/height(net1_res(:,13)));
    RMSE_SM11 = RMSE_SM11 + sqrt(sum((net1_res(:,14)).^2)/height(net1_res(:,14)));
    RMSE_SM12 = RMSE_SM12 + sqrt(sum((net1_res(:,15)).^2)/height(net1_res(:,15)));

end

RMSE_H = RMSE_H/10
RMSE_A = RMSE_A/10
RMSE_V = RMSE_V/10
RMSE_SM1 = RMSE_SM1/10
RMSE_SM2 = RMSE_SM2/10
RMSE_SM3 = RMSE_SM3/10
RMSE_SM4 = RMSE_SM4/10
RMSE_SM5 = RMSE_SM5/10
RMSE_SM6 = RMSE_SM6/10
RMSE_SM7 = RMSE_SM7/10
RMSE_SM8 = RMSE_SM8/10
RMSE_SM9 = RMSE_SM9/10
RMSE_SM10 = RMSE_SM10/10
RMSE_SM11 = RMSE_SM11/10
RMSE_SM12 = RMSE_SM12/10

%% ANN (My PCA SM with previous NaN, Only SM 1-5)

load('Nuc_ALL_Interphase_Cells_SM_1_to_5_Includes_Previous_NaN.mat')

RMSE_H = 0;
RMSE_A = 0;
RMSE_V = 0;
RMSE_SM1 = 0;
RMSE_SM2 = 0;
RMSE_SM3 = 0;
RMSE_SM4 = 0;
RMSE_SM5 = 0;

for k = 1:10
    net1 = models{k};
    net1_pred = net1(M_Inter(:,596:600)');
    net1_res = [M_Inter(:,[631:633,616:620])] - net1_pred';
    
    RMSE_H = RMSE_H + sqrt(sum((net1_res(:,1)).^2)/height(net1_res(:,1)))*1.6985;
    RMSE_A = RMSE_A + sqrt(sum((net1_res(:,2)).^2)/height(net1_res(:,2)))*107.82;
    RMSE_V = RMSE_V + sqrt(sum((net1_res(:,3)).^2)/height(net1_res(:,3)))*156.5;
    RMSE_SM1 = RMSE_SM1 + sqrt(sum((net1_res(:,4)).^2)/height(net1_res(:,4)));
    RMSE_SM2 = RMSE_SM2 + sqrt(sum((net1_res(:,5)).^2)/height(net1_res(:,5)));
    RMSE_SM3 = RMSE_SM3 + sqrt(sum((net1_res(:,6)).^2)/height(net1_res(:,6)));
    RMSE_SM4 = RMSE_SM4 + sqrt(sum((net1_res(:,7)).^2)/height(net1_res(:,7)));
    RMSE_SM5 = RMSE_SM5 + sqrt(sum((net1_res(:,8)).^2)/height(net1_res(:,8)));
    
end

RMSE_H = RMSE_H/10
RMSE_A = RMSE_A/10
RMSE_V = RMSE_V/10
RMSE_SM1 = RMSE_SM1/10
RMSE_SM2 = RMSE_SM2/10
RMSE_SM3 = RMSE_SM3/10
RMSE_SM4 = RMSE_SM4/10
RMSE_SM5 = RMSE_SM5/10

%% Interphase MLR ANOVA

load('MLRModels_INTERPHASE_ALL_Z_Scaled_Ver_2.mat')

anv_V = anova(Volume);
anv_V_exp_var = anv_V;
anv_V_exp_var(:,6) = array2table(100*(table2array(anv_V(1:end,1)))./sum(table2array(anv_V(1:end,1))));

anv_H = anova(Height);
anv_H_exp_var = anv_H;
anv_H_exp_var(:,6) = array2table(100*(table2array(anv_H(1:end,1)))./sum(table2array(anv_H(1:end,1))));

anv_A = anova(Area);
anv_A_exp_var = anv_A;
anv_A_exp_var(:,6) = array2table(100*(table2array(anv_A(1:end,1)))./sum(table2array(anv_A(1:end,1))));

%% Scatterplots (ANN)

% 1. NN (My PCA SM with Previous NaN)

load('Nuc_ALL_Interphase_Cells_ALL_Includes_Previous_NaN.mat')

Hp = [];
Ap = [];
Vp = [];
SM1p = [];
SM2p = [];
SM3p = [];
SM4p = [];
SM5p = [];
SM6p = [];
SM7p = [];
SM8p = [];
SM9p = [];
SM10p = [];
SM11p = [];
SM12p = [];

Ht = [];
At = [];
Vt = [];
SM1t = [];
SM2t = [];
SM3t = [];
SM4t = [];
SM5t = [];
SM6t = [];
SM7t = [];
SM8t = [];
SM9t = [];
SM10t = [];
SM11t = [];
SM12t = [];

for k = 1:10
    net1 = models{k};
    net1_pred = net1(M_Inter(:,[628:630,596:615])');
    net1_true = [M_Inter(:,[631:633,616:627])]';
    
    Hp(:,width(Hp)+1) = net1_pred(1,:)';
    Ht(:,width(Ht)+1) = net1_true(1,:)';
    Ap(:,width(Hp)+1) = net1_pred(2,:)';
    At(:,width(Ht)+1) = net1_true(2,:)';
    Vp(:,width(Hp)+1) = net1_pred(3,:)';
    Vt(:,width(Ht)+1) = net1_true(3,:)';
    SM1p(:,width(Hp)+1) = net1_pred(4,:)';
    SM1t(:,width(Ht)+1) = net1_true(4,:)';
    SM2p(:,width(Hp)+1) = net1_pred(5,:)';
    SM2t(:,width(Ht)+1) = net1_true(5,:)';
    SM3p(:,width(Hp)+1) = net1_pred(6,:)';
    SM3t(:,width(Ht)+1) = net1_true(6,:)';
    SM4p(:,width(Hp)+1) = net1_pred(7,:)';
    SM4t(:,width(Ht)+1) = net1_true(7,:)';
    SM5p(:,width(Hp)+1) = net1_pred(8,:)';
    SM5t(:,width(Ht)+1) = net1_true(8,:)';
    SM6p(:,width(Hp)+1) = net1_pred(9,:)';
    SM6t(:,width(Ht)+1) = net1_true(9,:)';
    SM7p(:,width(Hp)+1) = net1_pred(10,:)';
    SM7t(:,width(Ht)+1) = net1_true(10,:)';
    SM8p(:,width(Hp)+1) = net1_pred(11,:)';
    SM8t(:,width(Ht)+1) = net1_true(11,:)';
    SM9p(:,width(Hp)+1) = net1_pred(12,:)';
    SM9t(:,width(Ht)+1) = net1_true(12,:)';
    SM10p(:,width(Hp)+1) = net1_pred(13,:)';
    SM10t(:,width(Ht)+1) = net1_true(13,:)';
    SM11p(:,width(Hp)+1) = net1_pred(14,:)';
    SM11t(:,width(Ht)+1) = net1_true(14,:)';
    SM12p(:,width(Hp)+1) = net1_pred(15,:)';
    SM12t(:,width(Ht)+1) = net1_true(15,:)';

end

H = [mean(Ht')',mean(Hp')'];
A = [mean(At')',mean(Ap')'];
V = [mean(At')',mean(Ap')'];
SM1 = [mean(SM1t')',mean(SM1p')'];
SM2 = [mean(SM2t')',mean(SM2p')'];
SM3 = [mean(SM3t')',mean(SM3p')'];
SM4 = [mean(SM4t')',mean(SM4p')'];
SM5 = [mean(SM5t')',mean(SM5p')'];
SM6 = [mean(SM6t')',mean(SM6p')'];
SM7 = [mean(SM7t')',mean(SM7p')'];
SM8 = [mean(SM8t')',mean(SM8p')'];
SM9 = [mean(SM9t')',mean(SM9p')'];
SM10 = [mean(SM10t')',mean(SM10p')'];
SM11 = [mean(SM11t')',mean(SM11p')'];
SM12 = [mean(SM12t')',mean(SM12p')'];

H = (H*1.6985) + 6.8572;
A = (A*107.82) + 550.55;
V = (V*156.50) + 540.88;

figure
subplot(5,5,1)
scatter1 = scatter(H(:,1),H(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([2,17],[2,17],'k','LineWidth',1)
hold off
grid on
xlabel('Actual (\mum)')
ylabel('Predicted (\mum)')
title('Height')
xlim([2,17])
ylim([2,17])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,2)
scatter1 = scatter(A(:,1),A(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([100,1300],[100,1300],'k','LineWidth',1)
hold off
grid on
xlabel('Actual (\mum^2)')
ylabel('Predicted (\mum^2)')
title('Surface Area')
xlim([100,1300])
ylim([100,1300])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,3)
scatter1 = scatter(V(:,1),V(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([0,1600],[0,1600],'k','LineWidth',1)
hold off
grid on
xlabel('Actual (\mum^3)')
ylabel('Predicted (\mum^3)')
title('Volume')
xlim([0,1600])
ylim([0,1600])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,6)
scatter1 = scatter(SM1(:,1),SM1(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual')
ylabel('Predicted')
title('Shape Mode 1')
xlim([-6,6])
ylim([-6,6])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,7)
scatter1 = scatter(SM2(:,1),SM2(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual')
ylabel('Predicted')
title('Shape Mode 2')
xlim([-6,6])
ylim([-6,6])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,8)
scatter1 = scatter(SM3(:,1),SM3(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual')
ylabel('Predicted')
title('Shape Mode 3')
xlim([-6,6])
ylim([-6,6])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,11)
scatter1 = scatter(SM4(:,1),SM4(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual')
ylabel('Predicted')
title('Shape Mode 4')
xlim([-6,6])
ylim([-6,6])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,12)
scatter1 = scatter(SM5(:,1),SM5(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual')
ylabel('Predicted')
title('Shape Mode 5')
xlim([-6,6])
ylim([-6,6])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,13)
scatter1 = scatter(SM6(:,1),SM6(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual')
ylabel('Predicted')
title('Shape Mode 6')
xlim([-6,6])
ylim([-6,6])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,16)
scatter1 = scatter(SM7(:,1),SM7(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual')
ylabel('Predicted')
title('Shape Mode 7')
xlim([-6,6])
ylim([-6,6])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,17)
scatter1 = scatter(SM8(:,1),SM8(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual')
ylabel('Predicted')
title('Shape Mode 8')
xlim([-6,6])
ylim([-6,6])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,18)
scatter1 = scatter(SM9(:,1),SM9(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual')
ylabel('Predicted')
title('Shape Mode 9')
xlim([-6,6])
ylim([-6,6])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,21)
scatter1 = scatter(SM10(:,1),SM10(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual')
ylabel('Predicted')
title('Shape Mode 10')
xlim([-6,6])
ylim([-6,6])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,22)
scatter1 = scatter(SM11(:,1),SM11(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual')
ylabel('Predicted')
title('Shape Mode 11')
xlim([-6,6])
ylim([-6,6])
set(gca,'DataAspectRatio',[1 1 1])

subplot(5,5,23)
scatter1 = scatter(SM12(:,1),SM12(:,2),4,'filled','MarkerFaceColor','#375D6D');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual')
ylabel('Predicted')
title('Shape Mode 12')
xlim([-6,6])
ylim([-6,6])
set(gca,'DataAspectRatio',[1 1 1])

%% Scatterplots (MLR)

% 1. MLR (Interphase MLR ALL)

load('MLRModels_INTERPHASE_ALL_Z_Scaled_Ver_2.mat')

Hp = [];
Ap = [];
Vp = [];
SM1p = [];
SM2p = [];
SM3p = [];
SM4p = [];
SM5p = [];

Ht = [];
At = [];
Vt = [];
SM1t = [];
SM2t = [];
SM3t = [];
SM4t = [];
SM5t = [];

for k = 1:10
    net1 = models{k};
    net1_pred = net1(M_Inter(:,[628:630,596:615])');
    net1_true = [M_Inter(:,[631:633,616:627])]';
    
    Hp(:,width(Hp)+1) = net1_pred(1,:)';
    Ht(:,width(Ht)+1) = net1_true(1,:)';
    Ap(:,width(Hp)+1) = net1_pred(2,:)';
    At(:,width(Ht)+1) = net1_true(2,:)';
    Vp(:,width(Hp)+1) = net1_pred(3,:)';
    Vt(:,width(Ht)+1) = net1_true(3,:)';
    SM1p(:,width(Hp)+1) = net1_pred(4,:)';
    SM1t(:,width(Ht)+1) = net1_true(4,:)';
    SM2p(:,width(Hp)+1) = net1_pred(5,:)';
    SM2t(:,width(Ht)+1) = net1_true(5,:)';
    SM3p(:,width(Hp)+1) = net1_pred(6,:)';
    SM3t(:,width(Ht)+1) = net1_true(6,:)';
    SM4p(:,width(Hp)+1) = net1_pred(7,:)';
    SM4t(:,width(Ht)+1) = net1_true(7,:)';
    SM5p(:,width(Hp)+1) = net1_pred(8,:)';
    SM5t(:,width(Ht)+1) = net1_true(8,:)';

end

H = [mean(Ht')',mean(Hp')'];
A = [mean(At')',mean(Ap')'];
V = [mean(At')',mean(Ap')'];
SM1 = [mean(SM1t')',mean(SM1p')'];
SM2 = [mean(SM2t')',mean(SM2p')'];
SM3 = [mean(SM3t')',mean(SM3p')'];
SM4 = [mean(SM4t')',mean(SM4p')'];
SM5 = [mean(SM5t')',mean(SM5p')'];

H = (H*1.6985) + 6.8572;
A = (A*107.82) + 550.55;
V = (V*156.50) + 540.88;

figure
scatter1 = scatter(H(:,1),H(:,2),4,'b','filled');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([2,17],[2,17],'k','LineWidth',1)
hold off
grid on
xlabel('Actual Height (\mum)')
ylabel('Predicted Height (\mum)')
title('ANN Predicted vs. Actual Nuclear Height')
xlim([2,17])
ylim([2,17])

figure
scatter1 = scatter(A(:,1),A(:,2),4,'b','filled');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([100,1300],[100,1300],'k','LineWidth',1)
hold off
grid on
xlabel('Actual Area (\mum^2)')
ylabel('Predicted Area (\mum^2)')
title('ANN Predicted vs. Actual Nuclear Area')
xlim([100,1300])
ylim([100,1300])

figure
scatter1 = scatter(V(:,1),V(:,2),4,'b','filled');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([0,1600],[0,1600],'k','LineWidth',1)
hold off
grid on
xlabel('Actual Volume (\mum^3)')
ylabel('Predicted Volume (\mum^3)')
title('ANN Predicted vs. Actual Nuclear Volume')
xlim([0,1600])
ylim([0,1600])

figure
scatter1 = scatter(SM1(:,1),SM1(:,2),4,'b','filled');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-4,4],[-4,4],'k','LineWidth',1)
hold off
grid on
xlabel('Actual Shape Mode 1 (z-score)')
ylabel('Predicted Shape Mode 1 (z-score)')
title('ANN Predicted vs. Actual Nuclear Shape Mode 1')
xlim([-4,4])
ylim([-4,4])

figure
scatter1 = scatter(SM2(:,1),SM2(:,2),4,'b','filled');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual Shape Mode 2 (z-score)')
ylabel('Predicted Shape Mode 2 (z-score)')
title('ANN Predicted vs. Actual Nuclear Shape Mode 2')
xlim([-6,6])
ylim([-6,6])

figure
scatter1 = scatter(SM3(:,1),SM3(:,2),4,'b','filled');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual Shape Mode 3 (z-score)')
ylabel('Predicted Shape Mode 3 (z-score)')
title('ANN Predicted vs. Actual Nuclear Shape Mode 3')
xlim([-6,6])
ylim([-6,6])

figure
scatter1 = scatter(SM4(:,1),SM4(:,2),4,'b','filled');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual Shape Mode 4 (z-score)')
ylabel('Predicted Shape Mode 4 (z-score)')
title('ANN Predicted vs. Actual Nuclear Shape Mode 4')
xlim([-6,6])
ylim([-6,6])

figure
scatter1 = scatter(SM5(:,1),SM5(:,2),4,'b','filled');
scatter1.MarkerFaceAlpha = .15;
hold on
plot([-6,6],[-6,6],'k','LineWidth',1)
hold off
grid on
xlabel('Actual Shape Mode 5 (z-score)')
ylabel('Predicted Shape Mode 5 (z-score)')
title('ANN Predicted vs. Actual Nuclear Shape Mode 5')
xlim([-6,6])
ylim([-6,6])

