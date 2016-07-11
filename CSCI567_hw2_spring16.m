load toyGMM.mat

accuracy1 = trainEvalModels(dataTr, dataTe, true);

x1 = randperm(length(dataTr.x1));
x2 = randperm(length(dataTr.x2));
x3 = randperm(length(dataTr.x3));

trainData = struct('x1',dataTr.x1(x1(1:10),:),'x2',dataTr.x2(x2(1:10),:),'x3',dataTr.x3(x3(1:15),:));
accuracy2 = trainEvalModels(trainData, dataTe, false);

trainData = struct('x1',dataTr.x1(x1(1:50),:),'x2',dataTr.x2(x2(1:50),:),'x3',dataTr.x3(x3(1:75),:));
accuracy3 = trainEvalModels(trainData, dataTe, false);

trainData = struct('x1',dataTr.x1(x1(1:100),:),'x2',dataTr.x2(x2(1:100),:),'x3',dataTr.x3(x3(1:150),:));
accuracy4 = trainEvalModels(trainData, dataTe, false);

trainData = struct('x1',dataTr.x1(x1(1:250),:),'x2',dataTr.x2(x2(1:250),:),'x3',dataTr.x3(x3(1:375),:));
accuracy5 = trainEvalModels(trainData, dataTe, false);

trainData = struct('x1',dataTr.x1(x1(1:500),:),'x2',dataTr.x2(x2(1:500),:),'x3',dataTr.x3(x3(1:750),:));
accuracy6 = trainEvalModels(trainData, dataTe, false);

trainData = struct('x1',dataTr.x1(x1(1:1000),:),'x2',dataTr.x2(x2(1:1000),:),'x3',dataTr.x3(x3(1:1500),:));
accuracy7 = trainEvalModels(trainData, dataTe, false);

accuracy = [accuracy2;accuracy3;accuracy4;accuracy5;accuracy6;accuracy7];
percentage = [1,5,10,25,50,100];
plot(percentage', accuracy);title('4.2 (e)');xlabel('Percentage of training data');ylabel('Accuracy');legend('Model 1', 'Model 2', 'Model 3');

spam();