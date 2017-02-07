%Final Project 

%Author: Bernardo Gomes
%--------------------------------------------
%%

%beginning
clc;
clear all;
close all;
%--------------------------------------------
%Load Breast Cancer Wisconsin (Original) Data Set
data_BreastCancer=LoadBreastCancerData(  );
%----------------------------------------------------------
%%
%Feature selection
X=data_BreastCancer(:, 2:10);
Y=data_BreastCancer(:, 11);

%Compute entropy of each feature
H=zeros(1, size(X,2));%allocate vector for each dimension entropy
for i=1:1:size(X,2)
    x=transpose(X(:,i));
    [a b]=hist(x, unique(x));
    total=sum(a);
    probabilities=a./total;
    H(i)=entropy(probabilities,2);
    fprintf( 'The entropy of %s is %f\n', getFeature(i), H(i));
    figure
    stem(b,probabilities)
    title(['Probability mass function of ', getFeature(i)])
    ylabel('probability')
end

%Compute Mutual Information between each feature and the class feature
Mutual_information=zeros(1,size(X,2));
for i=1:1:size(X,2)
    x=X(:,i);
    y=Y;
    d=[x y];
    %get different row ocurrencies
    dif=unique(d, 'rows');
    %count the number of ocurrencies of a row
    counts=count_ocurrencies(dif, d);
    %calculate joint pmf
    total=sum(counts);
    probabilities=counts./total;
    %Calculation of mutual information
    Mutual_information(i)=Mutual_Information(probabilities, dif, X(:,i), Y); 
end


%%

%-----After-----------------
% x_train = X(1:478, :); 
% x_test = X(479:end, :);
% y_train = Y(1:478); 
% y_test = Y(479:end);
% ypred = classify(x_test, x_train, y_train);
% sum(y_test ~= ypred);
% f=@(x_train, y_train, x_test, y_test) sum(y_test~=classify(x_test, x_train, y_train));
% 
% count=zeros(1,9);
% for i=1:1:20
%     inmodel = sequentialfs(f,X,Y);
%     count=count+inmodel;
% end
% Final_inmodel=count>15;
% [string, columns]=SpecifyFeatures(Final_inmodel);
% disp(string);
% 
% X_meaningfull=X(:, columns);
% 
% %Inicialization of COOLCAT
% dif=unique(X_meaningfull, 'rows');
% %count the number of ocurrencies of a row
% counts=count_ocurrencies(dif, X_meaningfull);
% %calculate pmf
% total=sum(counts);
% probabilities=counts./total;            
% %get row indexes of starting points for cluster
% cluster_rows=estimate_two_clusters(dif, probabilities);
% 
% %initialize the clusters
% Cluster_1=dif(cluster_rows(1), :);
% Cluster_1_probabilities(1)=probabilities(cluster_rows(1));
% 
% Cluster_2=dif(cluster_rows(2), :);
% Cluster_2_probabilities(1)=probabilities(cluster_rows(2));
% %delete rows from the unprocessed points
% dif([cluster_rows(1) cluster_rows(2)], :)=[];
% probabilities([cluster_rows(1) cluster_rows(2)])=[];
% 
% %Select which cluster data should join
% while isempty(probabilities)==0
%     Cluster_1_probabilities=[probabilities(1) Cluster_1_probabilities];
%     Cluster_2_probabilities=[probabilities(1) Cluster_2_probabilities];
%     H1=entropy(Cluster_1_probabilities,2);
%     H2=entropy(Cluster_2_probabilities,2);
%     if H1<H2 %P1 belongs to cluster 1
%         Cluster_2_probabilities(1)=[];
%         Cluster_1=[dif(1, :);Cluster_1];
%         dif(1, :)=[];
%         probabilities(1)=[];
%     else %P1 belongs to cluster 2
%         Cluster_1_probabilities(1)=[];
%         Cluster_2=[dif(1, :);Cluster_2];
%         dif(1, :)=[];
%         probabilities(1)=[];
%     end
% 
% end
% 
% 
% %Evaluate clusters
% c1= data_BreastCancer(:, 11)==2;
% c2=data_BreastCancer(:,11)==4;
% true_cluster1=data_BreastCancer(c1,:);
% true_cluster1=true_cluster1(:,columns+1);
% true_cluster1=unique(true_cluster1, 'rows');
% true_cluster2=data_BreastCancer(c2,:);
% true_cluster2=true_cluster2(:,columns+1);
% true_cluster2=unique(true_cluster2, 'rows');