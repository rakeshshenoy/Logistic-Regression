function accuracy = trainEvalModels(dataTr, dataTe, flag)
    
    %% MLE learning of model1, Gaussian Discriminative Analysis I
    mx1 = mean(dataTr.x1);
    mx2 = mean(dataTr.x2);
    mx3 = mean(dataTr.x3);

    Sx1 = cov(dataTr.x1);
    Sx2 = cov(dataTr.x2);
    Sx3 = cov(dataTr.x3);
    
    pix1 = length(dataTr.x1)/(length(dataTr.x1) + length(dataTr.x2) + length(dataTr.x3));
    pix2 = length(dataTr.x2)/(length(dataTr.x1) + length(dataTr.x2) + length(dataTr.x3));
    pix3 = length(dataTr.x3)/(length(dataTr.x1) + length(dataTr.x2) + length(dataTr.x3));
    
    pi = [pix1,pix2,pix3];
    
    model1 = struct('pi',pi,'m1',mx1,'m2',mx2,'m3',mx3,'S1',Sx1,'S2',Sx2,'S3',Sx3);
    
    value1 = findValue(dataTe.x1,1,mx1,mx2,mx3,Sx1,Sx2,Sx3,pix1,pix2,pix3);
    value2 = findValue(dataTe.x2,2,mx1,mx2,mx3,Sx1,Sx2,Sx3,pix1,pix2,pix3);
    value3 = findValue(dataTe.x3,3,mx1,mx2,mx3,Sx1,Sx2,Sx3,pix1,pix2,pix3);
    
    accuracy1 = (value1 + value2 + value3)/8750*100;
    
    %% MLE learning of model2, Gaussian Discriminative Analysis II
    equalS = (length(dataTr.x1)*Sx1 + length(dataTr.x2)*Sx2 + length(dataTr.x3)*Sx3)/(length(dataTr.x1) + length(dataTr.x2) + length(dataTr.x3));
    
    model2 = struct('pi',pi,'m1',mx1,'m2',mx2,'m3',mx3,'S',equalS);
    
    value1 = findValue(dataTe.x1,1,mx1,mx2,mx3,equalS,equalS,equalS,pix1,pix2,pix3);
    value2 = findValue(dataTe.x2,2,mx1,mx2,mx3,equalS,equalS,equalS,pix1,pix2,pix3);
    value3 = findValue(dataTe.x3,3,mx1,mx2,mx3,equalS,equalS,equalS,pix1,pix2,pix3);
    accuracy2 = (value1 + value2 + value3)/8750*100;
    
    %% learning of model3, the MLR classifeir
    A = vertcat(dataTr.x1,dataTr.x2,dataTr.x3);
    B = vertcat(ones(length(dataTr.x1),1),ones(length(dataTr.x2),1)*2,ones(length(dataTr.x3),1)*3);
    C = mnrfit(A,B);
    value1 = findRegressionValue(dataTe.x1,1,C);
    value2 = findRegressionValue(dataTe.x2,2,C);
    value3 = findRegressionValue(dataTe.x3,3,C);
    accuracy3 = (value1 + value2 + value3)/8750*100;
    model3 = struct('w',[C zeros(size(C,1),1)]);
    
    %% visualize and compare learned models
    %if(flag == true)
        %plotBoarder(model1, model2, model3, dataTe);
    %end
    
    accuracy = [accuracy1, accuracy2, accuracy3];
    
end

function value = findValue(testDataset,dataSet,mx1,mx2,mx3,Sx1,Sx2,Sx3,pix1,pix2,pix3)
        value = 0;
        for k = 1:length(testDataset)
            val(k,1) = pix1*(1/(2*pi*det(Sx1)))*exp(-0.5*(testDataset(k,:)'-mx1')'*(inv(Sx1)*testDataset(k,:)'-mx1'));
            val(k,2) = pix2*(1/(2*pi*det(Sx2)))*exp(-0.5*(testDataset(k,:)'-mx2')'*(inv(Sx2)*testDataset(k,:)'-mx2'));
            val(k,3) = pix3*(1/(2*pi*det(Sx3)))*exp(-0.5*(testDataset(k,:)'-mx3')'*(inv(Sx3)*testDataset(k,:)'-mx3'));
         
            [~,i] = sort(val(k,:));
            
            if(i(3) == dataSet)
                value = value + 1;
            end
        end
end
    
function value = findRegressionValue(testDataset,dataSet,C)
    value = 0;
    for k = 1:length(testDataset)
        pihat = mnrval(C,testDataset(k,:));
        [~,i] = sort(pihat);
        
        if(i(3) == dataSet)
            value = value + 1;
        end
    end
end
