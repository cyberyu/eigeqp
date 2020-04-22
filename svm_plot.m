rng('default')
n = 100;
X = [2 + 0.5*randn(n,2);...
    3 + 0.6*randn(n,2)];
Y = [-ones(n,1);ones(n,1)];

subplot(2,3,1);

gscatter(X(:,1),X(:,2),Y);
title('Binary Toy Data');

x = linspace(0,5);
y = linspace(0,5);
[XX,YY] = meshgrid(x,y);

pred = [XX(:),YY(:)];

n = size(X,1);
m = size(pred,1);

K = zeros(n,n);
Km = zeros(n,m);

for loop=1:1:n
    K(:,loop)=RBF_kernel(X(loop,:),X,1);
    %K(:,loop)=lin_kernel(X(loop,:),X);
end



subplot(2,3,4);
imagesc(K);
title('RBF Kernel Matrix');

gamma = 20;

K = K + 1/gamma.*eye(size(K));

for loop=1:1:m
    Km(:,loop)=RBF_kernel(pred(loop,:),X,1);
    %Km(:,loop)=lin_kernel(pred(loop,:),X);
end

onevec = ones(n,1);

xopt = optimal_sol_lssvm(K,Y', onevec);

Ypd = zeros(1,n);
Ymespd = zeros(1,m);

for loop=1:1:n
    Ypd(loop) = sign(sum(K(:,loop).*Y.*xopt));
end

for loop=1:1:m
    Ymespd(loop) = sign(sum(Km(:,loop).*Y.*xopt));
end

subplot(2,3,2);

gscatter(X(:,1),X(:,2),Ypd);
title('Predictions Original');


pred = [XX(:),YY(:)];


subplot(2,3,3);
gscatter(pred(:,1),pred(:,2), Ymespd);
title('Decision Boundary Original');



[U,V]=svd(K);

k=10;

%         Uk = U(:,1:k);
%         Vk = V(1:k,1:k);
% 
%         Ck = diag(Y'.*1./(Y'*Uk));
        
C = diag(Y'.*1./(Y'*U));
Kproj = C*C*V;
onevecproj = C*U'*onevec;

xopt_proj = optimal_sol_lssvm(Kproj,Y',onevecproj);
xopt_recover = U*C*xopt_proj;


Ypd = zeros(1,n);
Ymespd = zeros(1,m);

for loop=1:1:n
    Ypd(loop) = sign(sum(K(:,loop).*Y.*xopt_recover));
end

subplot(2,3,5);

gscatter(X(:,1),X(:,2),Ypd);
title('Predictions Diagonalized');


for loop=1:1:m
    Ymespd(loop) = sign(sum(Km(:,loop).*Y.*xopt_recover));
end

subplot(2,3,6);
gscatter(pred(:,1),pred(:,2), Ymespd);
title('Decision Boundary Diagonalized');



