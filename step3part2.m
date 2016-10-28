load trainData.mat
data1 = X1; 
data2= X2;
data3=X3;
label = Y;

load testData.mat

test_data1 = X1;
test_data2=X2;
test_data3=X3;
test_label= Y;

trainData1 = data1;    testData1 = test_data1;
trainData2 = data2;    testData2 = test_data2;
trainData3 = data3;    testData3 = test_data3;
trainLabel = label;  testLabel = test_label;
numTrain = size(trainData1,1); numTest = size(testData1,1);

 K1 =  [ (1:numTrain)' , chi_square_kernel(trainData1,trainData1) ];
 KK1 = [ (1:numTest)'  , chi_square_kernel(testData1,trainData1)  ];
 K2 =  [ (1:numTrain)' , chi_square_kernel(trainData2,trainData2) ];
 KK2 = [ (1:numTest)'  , chi_square_kernel(testData2,trainData2)  ];
 K3 =  [ (1:numTrain)' , chi_square_kernel(trainData3,trainData3) ];
 KK3 = [ (1:numTest)'  , chi_square_kernel(testData3,trainData3)  ];
 
 model1 = svmtrain(trainLabel, K1, '-t 4 -c 10');
 model2 = svmtrain(trainLabel, K2, '-t 4 -c 10');
 model3 = svmtrain(trainLabel, K3, '-t 4 -c 10');
load step3kernel.mat

Ka=(K1+K2+K3)./3;

%Kb=(K1.*K2.*K3);
% Ktemp=bsxfun(@times,K1,K2);
% Kbt=bsxfun(@times,Ktemp,K3);
 %Kb=bsxfun(@power,Kbt,(1/3));
modela = svmtrain(trainLabel, Ka, '-t 4 -c 10');
 %modelb = svmtrain(trainLabel, Kb, '-t 4 -c 10 -h 0');

load KK.mat

KKa=(KK1+KK2+KK3)./3;
 %KKb=(KK1.*KK2.*KK3);
%KKtemp=bsxfun(@times,KK1,KK2);
 %KKbt=bsxfun(@times,KKtemp,KK3);
 %KKb=bsxfun(@power,KKbt,(1/3));

[predTestLabela, acca, decValsa] = svmpredict(testLabel, KKa, modela);

 
%[predTestLabelb, accb, decValsb] = svmpredict(testLabel, KKb, modelb);

