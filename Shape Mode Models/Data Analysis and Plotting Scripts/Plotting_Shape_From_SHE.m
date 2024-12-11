%% Import PCA SM Data

clear all
close all force
clc

load('SeparatedData.mat')
load('Nuc_ALL_Interphase_Cells_ALL_Includes_Previous_NaN.mat')
load('PCAMatrices.mat')
load('SMStats.mat')

net1 = models{9};
H_pred = net1(Interphase(:,[628:630,596:615])');
H_pred = H_pred(1,:)';

net1 = models{7};
A_pred = net1(Interphase(:,[628:630,596:615])');
A_pred = A_pred(2,:)';

net1 = models{7};
V_pred = net1(Interphase(:,[628:630,596:615])');
V_pred = V_pred(3,:)';

net1 = models{7};
SM1_pred = net1(Interphase(:,[628:630,596:615])');
SM1_pred = SM1_pred(4,:)';

net1 = models{9};
SM2_pred = net1(Interphase(:,[628:630,596:615])');
SM2_pred = SM2_pred(5,:)';

net1 = models{8};
SM3_pred = net1(Interphase(:,[628:630,596:615])');
SM3_pred = SM3_pred(6,:)';

net1 = models{6};
SM4_pred = net1(Interphase(:,[628:630,596:615])');
SM4_pred = SM4_pred(7,:)';

net1 = models{2};
SM5_pred = net1(Interphase(:,[628:630,596:615])');
SM5_pred = SM5_pred(8,:)';

net1 = models{10};
SM6_pred = net1(Interphase(:,[628:630,596:615])');
SM6_pred = SM6_pred(9,:)';

net1 = models{10};
SM7_pred = net1(Interphase(:,[628:630,596:615])');
SM7_pred = SM7_pred(10,:)';

net1 = models{3};
SM8_pred = net1(Interphase(:,[628:630,596:615])');
SM8_pred = SM8_pred(11,:)';

net1 = models{7};
SM9_pred = net1(Interphase(:,[628:630,596:615])');
SM9_pred = SM9_pred(12,:)';

net1 = models{9};
SM10_pred = net1(Interphase(:,[628:630,596:615])');
SM10_pred = SM10_pred(13,:)';

net1 = models{5};
SM11_pred = net1(Interphase(:,[628:630,596:615])');
SM11_pred = SM11_pred(14,:)';

net1 = models{5};
SM12_pred = net1(Interphase(:,[628:630,596:615])');
SM12_pred = SM12_pred(15,:)';

%% Reverse PCA and Z-Scoring to Reconstruct SHE Coefficients

SM1_pred = SM1_pred.*SM_nuc_Inter_std(1) + SM_nuc_Inter_mean(1);
SM2_pred = SM2_pred.*SM_nuc_Inter_std(2) + SM_nuc_Inter_mean(2);
SM3_pred = SM3_pred.*SM_nuc_Inter_std(3) + SM_nuc_Inter_mean(3);
SM4_pred = SM4_pred.*SM_nuc_Inter_std(4) + SM_nuc_Inter_mean(4);
SM5_pred = SM5_pred.*SM_nuc_Inter_std(5) + SM_nuc_Inter_mean(5);
SM6_pred = SM6_pred.*SM_nuc_Inter_std(6) + SM_nuc_Inter_mean(6);
SM7_pred = SM7_pred.*SM_nuc_Inter_std(7) + SM_nuc_Inter_mean(7);
SM8_pred = SM8_pred.*SM_nuc_Inter_std(8) + SM_nuc_Inter_mean(8);
SM9_pred = SM9_pred.*SM_nuc_Inter_std(9) + SM_nuc_Inter_mean(9);
SM10_pred = SM10_pred.*SM_nuc_Inter_std(10) + SM_nuc_Inter_mean(10);
SM11_pred = SM11_pred.*SM_nuc_Inter_std(11) + SM_nuc_Inter_mean(11);
SM12_pred = SM12_pred.*SM_nuc_Inter_std(12) + SM_nuc_Inter_mean(12);

SM_pred = [SM1_pred,SM2_pred,SM3_pred,SM4_pred,SM5_pred,SM6_pred,SM7_pred,SM8_pred,SM9_pred,SM10_pred,SM11_pred,SM12_pred];

SHE_pred = SM_pred*PCA_nuc_Inter(:,1:12)';

%% Apply the SH Functions on SHE Coefficients



%% Create Plot of 3D Shape


