function [ r ] = int2labelVec( y )

r = zeros(5,size(y,2));
for i = 1:max(y)
    r(i, y==i) = 1;
end

end

