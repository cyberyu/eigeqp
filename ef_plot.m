xacc =zeros(1,1000);
yacc =zeros(1,1000);
xaccproj = zeros(1,1000);
yaccproj = zeros(1,1000);
xaccrecover = zeros(1,1000);
yaccrecover = zeros(1,1000);
xaccfullrecover = zeros(1,1000);
yaccfullrecover = zeros(1,1000);

s = linspace(0.1,100,1000);

%LMPRXA20 = LMPRXA1(:,12:31);
LMPRXA20 = LMPRXA1;
LMPRXA20 = LMPRXA20 + 0.0001.*rand(size(LMPRXA20));
% a small test here
% generate a random matrix with 10 observations and 5 features
mu = mean(LMPRXA20,1);
% get the original covariance
%Q=cov(R') + 0.0001*eye(size(cov(R')));
Q=cov(LMPRXA20);
% apply eignvalue decomposition
[U,V]=svd(Q);
% calculate C
C = diag(1./sum(U,1));
% project R by left multiply C*H'
Rproj = C*U'*LMPRXA20';  
muproj = C*U'*mu';
% recalculate the covariance of projected data
Qproj = cov(Rproj');
% the resulting Qproj is a diagonal matrix equals to C*C*V
% Qproj = diag(diag(Qproj))+ 0.0001*eye(size(Q));
Qproj = diag(diag(Qproj));
% we assume we want the sum of x equals to 1, so 
A = ones(1,99);
b = 1;
lambda = 1;
mu = mu';

invQ=Q\eye(size(Q));
invQproj=Qproj\eye(size(Qproj));
    
for loop =1:1:1000
    scale = s(loop);
    
    % the original solution
    
    xopt = optimal_sol_inv(invQ,A,b,-mu,scale*lambda);
    px_risk = xopt'*Q*xopt;
    py_return = xopt'*mu;
    xacc(loop) = px_risk;
    yacc(loop) = py_return;
    
    % projection in factor space
    
    xoptproj = optimal_sol_inv(invQproj,A,b,-muproj,scale*lambda);
    px_risk_proj = xoptproj'*Qproj*xoptproj;
    py_return_proj = xoptproj'*muproj;   
    xaccproj(loop) = px_risk_proj;
    yaccproj(loop) = py_return_proj;
    
    % projection with scaling and rotation
    size(U*C*xoptproj)
    xfullrecover = U*C*xoptproj;
    px_risk_full_recover = xfullrecover'*Q*xfullrecover;
    py_return_full_recover = xfullrecover'*mu;    
    xaccfullrecover(loop) = px_risk_full_recover;
    yaccfullrecover(loop) = py_return_full_recover;
    
    % projection with rotation only
    xrecover = U*xoptproj;
    px_risk_recover = xrecover'*Q*xrecover;
    py_return_recover = xrecover'*mu;
    xaccrecover(loop) = px_risk_recover;
    yaccrecover(loop) = py_return_recover;    
    


end

subplot(2,2,1);
scatter(xacc,yacc,'bo');
title('EF in original space');
xlabel('Risk');
ylabel('Return');

subplot(2,2,2);
scatter(xaccproj,yaccproj,'ro');
title('EF in full factor space');
xlabel('Risk');
ylabel('Return');

subplot(2,2,3);
scatter(xaccfullrecover,yaccfullrecover,'ko');
title('EF in reconstructed space');
xlabel('Risk');
ylabel('Return');

subplot(2,2,4);
scatter(xaccrecover,yaccrecover,'go');
title('EF in rotated factor space');
xlabel('Risk');
ylabel('Return');


% hold on;


