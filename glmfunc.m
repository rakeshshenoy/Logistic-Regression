function glmfunc(trainData, trainLabel, testData, testLabel)
    p = glmfit(trainData, trainLabel, 'normal');
    yhatTrain = glmval(p,trainData,'identity');
    yhatTest = glmval(p,testData,'identity');
    
    for i=1:2300
           if  yhatTest(i)>=0.5
               yhatTest(i)=1;
           else
               yhatTest(i)=0;
           end
           if  yhatTrain(i)>=0.5
               yhatTrain(i)=1;
           else
               yhatTrain(i)=0;
           end
    end
    
    testCount = 0;
    trainCount = 0;

    for i=1:2300
        if yhatTest(i)==testLabel(i)
            testCount=testCount+1;
        end
    end
    testAcc = (testCount/2300)*100

    for i=1:2300
        if yhatTrain(i)==trainLabel(i)
            trainCount=trainCount+1;
        end
    end
    trainAcc = (trainCount/2300)*100
 
end