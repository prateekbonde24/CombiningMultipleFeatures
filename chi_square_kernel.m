function D = chi_square_kernel(X,Y)
D = zeros(size(X,1),size(Y,1));
for i=1:size(X,1)
    d=bsxfun(@minus, X(i,:), Y);
    s=bsxfun(@plus, X(i,:), Y);
    D(i,:)=sum(d.^2./(s/2+eps),2);
end
D=1-D;
end
