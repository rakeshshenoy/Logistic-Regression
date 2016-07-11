function normalized = normalizeData(data)
    mean1 = mean(data);
    std1 = std(data);
    normalized = zeros(length(data),size(data,2));
    for i = 1:length(data)
        normalized(i,1:size(data,2)) = (data(i,1:size(data,2)) - mean1)./ std1;
    end
end

