function spam()
    A = dlmread('spambase.data');
    A=A(randperm(length(A)),:);
    trainData(1:2300,1:57) = A(1:2300,1:57);
    trainLabel(1:2300,1:1) = A(1:2300,58:58);
    testData(1:2300,1:57) = A(2301:4600,1:57);
    testLabel(1:2300,1:1) = A(2301:4600,58:58);

    batchGradient(trainData, trainLabel, testData, testLabel);
    newtonsMethod(trainData, trainLabel, testData, testLabel);

    normTrainData = normalizeData(trainData);
    normTestData = normalizeData(testData);

    batchGradient(normTrainData, trainLabel, normTestData, testLabel);
    newtonsMethod(normTrainData, trainLabel, normTestData, testLabel);

    glmfunc(trainData, trainLabel, testData, testLabel);
    glmfunc(normTrainData, trainLabel, normTestData, testLabel);
end

