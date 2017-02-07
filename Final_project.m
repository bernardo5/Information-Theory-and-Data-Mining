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
fprintf( '-------------------------------------------------------------\n\n');

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
ordered_features=[[1;2;3;4;5;6;7;8;9] Mutual_information'];
ordered_features=sortrows(ordered_features,-2);
disp(string_oneFeature(ordered_features));
fprintf('--------------------------------------------------------------\n\n');

%Compute Mutual information between two features and the class feature
Mutual_information_2=zeros(size(X,2),size(X,2));
max_2=0;
ind=zeros(1,2);
for i=1:1:(size(X,2)-1)
    for j=i+1:1:size(X,2)
        
            x=X(:,i);
            y=X(:,j);
            d=[x y Y];
            %get different row ocurrencies
            dif=unique(d, 'rows');
            s=size(dif);
            %count the number of ocurrencies of a row
            counts=count_ocurrencies(dif, d);
            %calculate joint pmf
            total=sum(counts);
            probabilities=counts./total;
            
            %calculation of mutual information
            Mutual_information_2(i,j)=Mutual_Information_multi(probabilities, dif, [X(:,i) X(:,j)], Y);
            if max_2< Mutual_information_2(i,j)
               max_2= Mutual_information_2(i,j);
               ind(1)=i;
               ind(2)=j;
            end
    end
end
fprintf('The most relevant association of 2 features is %s with %s with a mutual information of %f\n', getFeature(ind(1)), getFeature(ind(2)), max_2);
fprintf('-----------------------------------------------------------\n\n');

%Compute Mutual information between three features and the class feature
max_3=0;
ind_3=zeros(1,3);
for i=1:1:size(X,2)
    for j=1:1:size(X,2)
        for k=1:1:size(X,2)
            if i~=j && i~=k && j~=k
                x=X(:,i);
                y=X(:,j);
                third=X(:,k);
                d=[x y third Y];
                %get different row ocurrencies
                dif=unique(d, 'rows');
                s=size(dif);
                %count the number of ocurrencies of a row
                counts=count_ocurrencies(dif, d);
                %calculate joint pmf
                total=sum(counts);
                probabilities=counts./total;

                %calculation of mutual information
                Mutual_information_3=Mutual_Information_multi(probabilities, dif, [X(:,i) X(:,j) X(:,k)], Y);
                if max_3< Mutual_information_3
                   max_3= Mutual_information_3;
                   ind_3(1)=i;
                   ind_3(2)=j;
                   ind_3(3)=k;
                end
            end
        end    
     end
end
fprintf('The most relevant association of 3 features is %s with %s and %s with a mutual information of %f\n', getFeature(ind_3(1)), getFeature(ind_3(2)), getFeature(ind_3(3)), max_3);
fprintf('-----------------------------------------------------------\n\n');

%Compute Mutual information between four features and the class feature
max_4=0;
ind_4=zeros(1,4);
for i=1:1:size(X,2)
    for j=1:1:size(X,2)
        for k=1:1:size(X,2)
            for z=1:1:size(X,2)
                if i~=j && i~=k && i~=z && j~=k && j~=z
                    x=X(:,i);
                    y=X(:,j);
                    third=X(:,k);
                    fourth=X(:,z);
                    d=[x y third fourth Y];
                    %get different row ocurrencies
                    dif=unique(d, 'rows');
                    s=size(dif);
                    %count the number of ocurrencies of a row
                    counts=count_ocurrencies(dif, d);
                    %calculate joint pmf
                    total=sum(counts);
                    probabilities=counts./total;

                    %calculation of mutual information
                    Mutual_information_4=Mutual_Information_multi(probabilities, dif, [X(:,i) X(:,j) X(:,k) X(:,z)], Y);
                    if max_4< Mutual_information_4
                       max_4= Mutual_information_4;
                       ind_4(1)=i;
                       ind_4(2)=j;
                       ind_4(3)=k;
                       ind_4(4)=z;
                    end
                end
            end
        end    
     end
end
fprintf('The most relevant association of 4 features is %s with %s and %s and %s with a mutual information of %f\n', getFeature(ind_4(1)), getFeature(ind_4(2)), getFeature(ind_4(3)), getFeature(ind_4(4)), max_4);
fprintf('-----------------------------------------------------------\n\n');
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