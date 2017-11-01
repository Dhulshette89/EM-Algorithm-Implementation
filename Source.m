clear all; close all; 
% X is the dataset size 100x2 
%   Column 1: sepal length %   Column 2: sepal width 
X = dlmread('simple_iris_dataset.dat');  % Size=100x2 
RandData=X(randperm(100),:);
N = length(X);  % N=100 
 % Initialization - take 2 random samples from data set 
 ctr1 = X(randi([1,N]),:);
 ctr2 = X(randi([1,N]),:); 
 cov1 = cov(X); cov2 = cov(X); 
 prior1 = 0.5; 
 prior2 = 0.5; 
 % Misc. initialization 
 idx_c1 = zeros(50,1); 
 idx_c2 = zeros(50,1); 
 W1 = zeros (100,1); 
 W2 = zeros (100,1); 
 mu=mean(X);
 idx1 = randperm(size(RandData,1),1);
 idx2 = randperm(size(RandData,1),1);
 mu1 = RandData(idx1,:);
 mu2 = RandData(idx2,:);
 M = repmat(mu,N,1);
 for itr = 1:250
   den = (mvnpdf(X,mu1,cov1)*prior1+mvnpdf(X,mu2,cov2)*prior2);
   W1=(mvnpdf(X,mu1,cov1)*prior1);
   W2=(mvnpdf(X,mu2,cov2)*prior2);
   W1 = W1./den;
   W2 = W2./den;
   prior1=mean(W1);
   prior2=mean(W2);
   ctr1=[W1 W1].*X/sum(W1);
   ctr2=[W2 W2].*X/sum(W2);
   mu1 = mean(ctr1);
   mu2 = mean(ctr2);
   M1 = repmat(mu1,N,1);
   M2 = repmat(mu2,N,1);
   cov1=(transpose([W1 W1].*(X-M1))*(X-M1))./sum(W1);
   cov2=(transpose([W2 W2].*(X-M2))*(X-M2))./sum(W2);
 end
 figure; hold on;  
 title('Clustering with EM algorithm'); 
 xlabel('Sepal Length'); 
 ylabel('Sepal Width'); 
 % Hard clustering assignment – W1, W2 (100x1) 
 idx_c1 = find(W1 > W2); 
 idx_c2 = find(W1 <= W2); 
 % idx_c1 is a vector containing the indices of the points in X  
 % that belong to cluster 1 (Mx1) 
 % idx_c2 is a vector containing the indices of the points in X  
 % that belong to cluster 2 (N-M x 1) 
 % Plot clustered data with two different colors 
 plot(X(idx_c1,1),X(idx_c1,2),'r.','MarkerSize',12) 
 plot(X(idx_c2,1),X(idx_c2,2),'b.','MarkerSize',10) 
 
 ctr1 = [mean(X(idx_c1,1)) mean(X(idx_c1,2))];
 ctr2 = [mean(X(idx_c2,1)) mean(X(idx_c2,2))];
% Plot centroid of each cluster – ctr1, ctr2  (1x2) 
plot(ctr1(:,1),ctr1(:,2), 'kx', 'MarkerSize',12,'LineWidth',2); 
plot(ctr2(:,1),ctr2(:,2), 'ko', 'MarkerSize',12,'LineWidth',2); 