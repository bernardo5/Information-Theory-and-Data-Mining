function [ cluster_entropy ] = Calc_Cluster_Entropy( cluster )
%based on the cluster matrix determines its entropy

dif=unique(cluster, 'rows');
%count the number of ocurrencies of a row
counts=count_ocurrencies(dif, cluster);
%calculate pmf
total=sum(counts);
probabilities=counts./total; 

cluster_entropy=entropy(probabilities,2);


end

