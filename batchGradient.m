function batchGradient(XTrain, yTrain, XTest, yTest)
    [~, nfeatures] = size(XTrain);
    w0 = rand(nfeatures + 1, 1);
    weight = logisticRegressionWeights( XTrain, yTrain, w0, 500, 0.9);
    trainRes = logisticRegressionClassify( XTrain, weight );
    res = logisticRegressionClassify( XTest, weight );
    trainErrors = abs(yTrain - trainRes);
    err1 = sum(trainErrors);
    errors = abs(yTest - res);
    err = sum(errors);
    trainAcc = 1 - err1 / size(XTrain, 1)
    testAcc = 1 - err / size(XTest, 1)
end

function w = logisticRegressionWeights(XTrain, yTrain, w0, repetitions, rate)
    [nSamples, nFeature] = size(XTrain);
    w = w0;
    precost = 0;
    acc = zeros(1,repetitions);
    for j = 1:repetitions
        gradient = zeros(nFeature + 1,1);
        for k = 1:nSamples
            gradient = gradient + (sigmoid([1.0 XTrain(k,:)] * w) - yTrain(k)) * [1.0 XTrain(k,:)]';
        end
        w = w - rate * gradient;
        cost = CostFunc(XTrain, yTrain, w);
        if j~=0 && abs(cost - precost) / cost <= 0.0001
            break;
        end
        precost = cost;
        res = logisticRegressionClassify(XTrain, w);
        errors = abs(yTrain - res);
        err = sum(errors);
        acc(j) = (1 - err / size(XTrain, 1))*100;
    end
    plot(1:repetitions,acc);title('Batch Gradient Descent');xlabel('Number of iterations');ylabel('Accuracy');
end

function [ res ] = logisticRegressionClassify( X, w )
    nTest = size(X,1);
    res = zeros(nTest,1);
    for i = 1:nTest
        sigm = sigmoid([1.0 X(i,:)] * w);
        if sigm >= 0.5
            res(i) = 1;
        else
            res(i) = 0;
        end
    end
end


