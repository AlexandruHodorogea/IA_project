function [ Xtr, ytr, Xv, yv ] = cv_split( x, y, ratio )

rp = randperm(size(x, 2));

v_indexes_num = size(x,2) * ratio;

Xtr = x(:,1:size(x,2) - v_indexes_num);
ytr = y(:,1:size(x,2) - v_indexes_num);

Xv = x(:,size(x,2) - v_indexes_num + 1:end);
yv = y(:,size(x,2) - v_indexes_num + 1:end);



end

