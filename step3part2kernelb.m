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

Kb=(K1.*K2.*K3).^(1/3);
Kb=Kb(1:4786,2:4787);
Kb=[(1:4786)',Kb];
model_precomputed= svmtrain(trainLabel,Kb,'-t 4 -c 10');


KKb=(KK1.*KK2.*KK3).^(1/3);
KKb=KKb(1:1883,2:1884);
KKb=[(1:1883)',KKb];
[predicted_st3_3, accuracy_st3_3, Prob_estimate_st3_3]=svmpredict(testLabel, KKb, model_precomputed);