load trainData.mat

 model1=svmtrain (Y,X1,'-c 10 -t 0 -b 1');
 model2=svmtrain (Y,X2,'-c 10 -t 0 -b 1');
 model3=svmtrain (Y,X3,'-c 10 -t 0 -b 1');

load testData.mat 
 [predicted_label1, accuracy1, Prob_estimate1]=svmpredict(Y,X1,model1,'-b 1');
[predicted_label2, accuracy2, Prob_estimate2]=svmpredict(Y,X2,model2,'-b 1');
 [predicted_label3, accuracy3, Prob_estimate3]=svmpredict(Y,X3,model3,'-b 1');

%load step0part2output.mat

sum1=(Prob_estimate1+Prob_estimate2+Prob_estimate3)/3;

[value,I]= max(sum1,[],2);
result=I==Y;
new=sum(result);
[total,ab]=size(Y);
accuracy= 100*(new/total);
display(accuracy);