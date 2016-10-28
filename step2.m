load trainData.mat

C=[X1,X2,X3];

model1=svmtrain (Y,C,'-c 10 -t 0 -b 1');


load testData.mat 


C_T=[X1,X2,X3];


[predicted_label1, accuracy1, Prob_estimates1]=svmpredict(Y,C_T,model1,'-b 1');