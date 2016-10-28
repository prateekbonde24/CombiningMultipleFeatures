load trainData.mat

K=[(1:4786)',rbfkernel(X1,1)];
model1 = svmtrain(Y,K,'-t 4 -c 40');


load testData.mat
KK1=[(1:1883)',rbfkernel(X1,1)];
[predicted_st3_1, accuracy_st3_1, Prob_estimate_st3_1]=svmpredict(Y, KK1, model1);
