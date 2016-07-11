function [ J ] = CostFunc( XTrain, yTrain, w )
    [nSamples, ~] = size(XTrain);
    temp = 0.0;
    for m = 1:nSamples
        hx = sigmoid([1.0 XTrain(m,:)] * w);
        if yTrain(m) == 1
            temp = temp + log(hx);
        else
            temp = temp + log(1 - hx);
        end
    end
    J = temp / (-nSamples);
end