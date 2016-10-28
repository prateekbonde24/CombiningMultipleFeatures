function z=rbfkernel(data,gamma)

D = squareform( pdist(data, 'euclidean') );
z= exp(-(D .^ 2) ./ ( 2*gamma^2));

end