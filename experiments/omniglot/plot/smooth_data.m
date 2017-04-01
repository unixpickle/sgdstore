function [smoothed] = smooth_data(data)
  rolling = mean(data(1:10));
  smoothed = zeros(size(data));
  for i=1:size(data)
    rolling = rolling + 0.05*(data(i)-rolling);
    smoothed(i) = rolling;
  end
end
