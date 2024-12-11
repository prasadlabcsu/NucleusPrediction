%% Loading Data

%Using 10-fold cross validation to predict Nuclear parameters from Cell

%Assuming all the data is in a mat file, just load it with your appropriate path
%load ..path/data.mat

load('SeparatedData.mat') % Loads the Interphase, Non-Interphase, Edge, and NaN cell datasets

CellData = Interphase(:,[596:600,603,604,606,608,611,612]); % Cell membrane volume (fL) and the 8 principal shape coefficients
NucData = Interphase(:,[631:633,616:627]); % Nuclear volume (fL)

%Pick the dependent variable. Note that if data is in columns you should transpose. Else remove the transpose
% assume that data.mat has two arrays: NucData and CellData

X = CellData';
t = NucData'; %Nuclear volume or aspect ratio or solidity or whatever

modref='Nuc_ALL_'; %label the models with the target variable. Here its Nuc volume

exvar = 'Interphase_Cells_My_Shape_Modes_Minus'; % Cell population is Interphase cells


kfold = 10;
Nsize = size(t,2);
fold=cvpartition(Nsize,'kfold',kfold); %Makes 10 partitions of the data. 
% Each partition has 10% in a test set and 90% in a training set.
%The indices of the classes are stored and will be called below.

fprintf('Data Loaded!')

%% Creating NN Model

% Choose a Training Function
% For a list of all training functions type: help nntrain
% 'trainlm' is usually fastest.
% 'trainbr' takes longer but may be better for challenging problems.
% 'trainscg' uses less memory. Suitable in low memory situations.

trainFcn = 'trainbr';  % Bayesian Regularization backpropagation.

% Size of the Neural Network This is a single layer network. 

%hiddenLayerSize = 20; 

%For two hidden layers replace by
hiddenLayerSize = [15, 15, 15];

%% Running NN Models

for i=1:kfold
    trainIdx = fold.training(i); %Returns a logical vector 
    % with 1 corresponding to the training set of that fold.
    testIdx = fold.test(i); %Logical vector of test partition
    xtrain = X(:,trainIdx); %picks out training set explanatory variables
    ytrain = t(:,trainIdx); %picks out training set target
    %Set up NN regression model
    net1 = fitnet(hiddenLayerSize,trainFcn); % or fitnet([hiddenLayer1Size hiddenLayer2Size],  trainFcn);
    %Declare the division of data between train and validation. 
    %Test is zero below since we are holding out the test set. 
    net1.divideParam.trainRatio = 85/100; 
    net1.divideParam.valRatio = 15/100;
    net1.divideParam.testRatio = 0/100; 
    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net1.performFcn = 'mse';    % Mean Squared Error
    net1.trainParam.epochs= 100; % Number of Epochs
    %Now train the model with the training data
    net1 = train(net1, xtrain, ytrain); %[net tr]=train(..) also returns a training record
    models{i} = net1; %store the trained model
    %Form the test data set
    xtest = X(:,testIdx); %test set explanatory variables
    ytest = t(:,testIdx); %test set target
    pred = net1(xtest);  %makes a prediction from the current model on test set
    Per = perform(net1,ytest,pred); %performance of the prediction (mse or whatever you picked)
    [Crr,pval] = corr(pred',ytest'); %correlation between prediction and actual.. more useful than per
    %Now save the performance data
    Performance(i,1) = Per; 
    Performance(i,2:(size(t,1)+1)) = diag(Crr)';
    Performance(i,(size(t,1)+2):(2*size(t,1)+1)) = diag(pval)';
end
nntraintool('close');

%% Saving Results in Output File

%Save results in an output folder
%FoldEr = 'path'; 

%make outful file names and save

filN = strcat(modref,exvar); %CHANGE FILE NAME

 filename = strcat(filN,'.mat');
 save(filename,'models');
 csvfilename = strcat(filN,'_Perf.csv');
csvwrite(csvfilename,Performance)
