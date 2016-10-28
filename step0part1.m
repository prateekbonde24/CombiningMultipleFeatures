load trainData.mat 
 
model1=svmtrain (Y,X1,'-c 10 -t 0 ');
 model2=svmtrain (Y,X2,'-c 10 -t 0 ');
model3=svmtrain (Y,X3,'-c 10 -t 0 ');

load testData.mat

[predicted_label1, accuracy1, Prob_estimate1]=svmpredict(Y,X1,model1);
[predicted_label2, accuracy2, Prob_estimate2]=svmpredict(Y,X2,model2);
[predicted_label3, accuracy3, Prob_estimate3]=svmpredict(Y,X3,model3);


