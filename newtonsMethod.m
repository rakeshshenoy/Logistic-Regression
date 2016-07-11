function newtonsMethod(XTrain, yTrain, XTest, yTest)
    [~, nfeatures] = size(XTrain);
    w0 = rand(nfeatures + 1, 1);
    weight = logisticRegressionNewtons( XTrain, yTrain, w0, 500);
    trainRes = logisticRegressionClassify( XTrain, weight );
    res = logisticRegressionClassify( XTest, weight );
    trainErrors = abs(yTrain - trainRes);
    err1 = sum(trainErrors);
    errors = abs(yTest - res);
    err = sum(errors);
    trainAcc = 1 - err1 / size(XTrain, 1)
    testAcc = 1 - err / size(XTest, 1)
end

function w = logisticRegressionNewtons(XTrain, yTrain, w0, repetitions)
    [nSamples, nFeature] = size(XTrain);
    w = w0;
    precost = 0;
    acc = zeros(1,repetitions);
    for j = 1:repetitions
        H = zeros(nFeature + 1,nFeature + 1);
        gradient = zeros(nFeature + 1,1);
        for k = 1:nSamples
            H = H - (sigmoid([1.0 XTrain(k,:)] * w) * (1 - sigmoid([1.0 XTrain(k,:)] * w))) * [1.0 XTrain(k,:)]' * [1.0 XTrain(k,:)];
            gradient = gradient + (sigmoid([1.0 XTrain(k,:)] * w) - yTrain(k)) * [1.0 XTrain(k,:)]';
            %display(size([1.0 XTrain(k,:)]));
        end
        A = inv(H + eye(nFeature + 1)*0.01);
        w = w - (A * gradient);
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
    display(acc);
    plot(1:repetitions,acc);title('Newton''s Method for Normalized Data');xlabel('Number of iterations');ylabel('Accuracy');
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

