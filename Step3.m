
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
%# test on testing data
[predTestLabel1, acc1, decVals1] = svmpredict(testLabel, KK1, model1);
[predTestLabel2, acc2, decVals2] = svmpredict(testLabel, KK2, model2);
[predTestLabel3, acc3, decVals3] = svmpredict(testLabel, KK3, model3);
  

%# test on training data
[predTrainLabel11, acc11, decVals11] = svmpredict(trainLabel, K1, model1);
[predTrainLabel22, acc22, decVals22] = svmpredict(trainLabel, K2, model2);
[predTrainLabel33, acc33, decVals33] = svmpredict(trainLabel, K3, model3);