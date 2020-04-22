xacc =zeros(1,1000);
yacc =zeros(1,1000);

xaccproj = zeros(10,1000);
yaccproj = zeros(10,1000);

xaccfull=zeros(1,1000);
yaccfull=zeros(1,1000);

xacck=zeros(10,1000);
yacck=zeros(10,1000);

xacck_rotation_only=zeros(10,1000);
yacck_rotation_only=zeros(10,1000);

s = linspace(0.1,100,1000);

%LMPRXA20 = LMPRXA1(:,12:31);
LMPRXA20 = LMPRXA1;
%LMPRXA20 = LMPRXA20 + 0.0001.*rand(size(LMPRXA20));
LMPRXA20 = LMPRXA20;
% a small test here
% generate a random matrix with 10 observations and 5 features
mu = mean(LMPRXA20,1);
% get the original covariance
%Q=cov(R') + 0.0001*eye(size(cov(R')));
Q=cov(LMPRXA20);
% apply eignvalue decomposition
[U,V]=svd(Q);


% we assume we want the sum of x equals to 1, so 
A = ones(1,99);
b = 1;
lambda = 1;
mu = mu';

%invQ=Q\eye(size(Q));
invQ = pinv(Q);

    
for loop =1:1:1000
    scale = s(loop);
    
    % the original solution
    
    xopt = optimal_sol_inv(invQ,A,b,-mu,scale*lambda);
    px_risk = xopt'*Q*xopt;
    py_return = xopt'*mu;
    xacc(loop) = px_risk;
    yacc(loop) = py_return;
    
    % projection in factor space
    for k = 5:10:95
        Uk = U(:,1:k);
        Vk = V(1:k,1:k);

        C = diag(1./sum(Uk,1));
        %C = diag(Y'.*1./(Y'*U));
        % project R by left multiply C*H'
        Rproj = C*Uk'*LMPRXA20';  
        muproj = C*Uk'*mu;
        % recalculate the covariance of projected data
        Qproj = cov(Rproj');
        % the resulting Qproj is a diagonal matrix equals to C*C*V
        % Qproj = diag(diag(Qproj))+ 0.0001*eye(size(Q));
        Qproj = diag(diag(Qproj));
        
        Ak = ones(1,k);
        %invQproj=Qproj\eye(size(Qproj));
        invQproj=pinv(Qproj);
        
        xoptproj = optimal_sol_inv(invQproj,Ak,b,-muproj,scale*lambda);
        
        px_risk_proj = xoptproj'*Qproj*xoptproj;
        py_return_proj = xoptproj'*muproj;   
        xaccproj((k+5)/10,loop) = px_risk_proj;
        yaccproj((k+5)/10,loop) = py_return_proj;   
        

        x_k_recover = Uk*C*xoptproj;
        px_risk_k_recover = x_k_recover'*Q*x_k_recover;
        py_return_k_recover = x_k_recover'*mu;    
        xacck((k+5)/10,loop) = px_risk_k_recover;
        yacck((k+5)/10,loop) = py_return_k_recover;        
        
        x_k_recover_rotation_only = Uk*xoptproj;
        px_risk_k_rotation_only = x_k_recover_rotation_only'*Q*x_k_recover_rotation_only;
        py_return_k_rotation_only = x_k_recover_rotation_only'*mu;    
        xacck_rotation_only((k+5)/10,loop) = px_risk_k_rotation_only;
        yacck_rotation_only((k+5)/10,loop) = py_return_k_rotation_only;  
    end

    %C = diag(1./sum(U,1));
    A = ones(1,99);
    C = diag(A*1./(A*U));
    Rproj = C*U'*LMPRXA20';  
    muproj = C*U'*mu;
    % recalculate the covariance of projected data
    Qproj = cov(Rproj');
    % the resulting Qproj is a diagonal matrix equals to C*C*V
    % Qproj = diag(diag(Qproj))+ 0.0001*eye(size(Q));
    Qproj = diag(diag(Qproj));
    
    % we assume we want the sum of x equals to 1, so 
    
    %invQproj=Qproj\eye(size(Qproj));
    invQproj=pinv(Qproj);
    
    xoptproj = optimal_sol_inv(invQproj,A,b,-muproj,scale*lambda);
    px_risk_proj = xoptproj'*Qproj*xoptproj;
    py_return_proj = xoptproj'*muproj;   
    xaccfull(loop) = px_risk_proj;
    yaccfull(loop) = py_return_proj;
    
end

subplot(1,3,1);
scatter(xacc,yacc,'k.');
title('EF in original space');
xlabel('Risk');
ylabel('Return');

subplot(1,3,2);

c = parula(10);

for k=1:1:10
    name = ['k=', num2str((k-1)*10+5)]; 
    plot(xaccproj(k,:),yaccproj(k,:),'Color',c(k,:),'LineWidth', 2, 'DisplayName', name);
    hold on;
end
plot(xaccfull(1:1000),yaccfull(1:1000),'Color','k','LineWidth', 2, 'DisplayName', 'Full');
h = legend('show','location','SouthEast');
title('EF in factor space');
xlabel('Risk');
ylabel('Return');

subplot(1,3,3);

for k=1:1:10
    name = ['k=', num2str((k-1)*10+5)]; 
    plot(xacck(k,:),yacck(k,:),'Color',c(k,:), 'LineWidth', 2,'DisplayName', name);
    hold on;
end
plot(xaccfull(1:1000),yaccfull(1:1000),'Color','k','LineWidth', 2,'DisplayName', 'Full');

h = legend('show','location','SouthEast');
title('EF recovered from factor space');
xlabel('Risk');
ylabel('Return');

% 
% subplot(2,2,4);
% 
% 
% for k=1:1:10
%     name = ['k=', num2str((k-1)*10+5)]; 
%     plot(xacck_rotation_only(k,:),yacck_rotation_only(k,:),'r.','MarkerSize', k/2, 'DisplayName', name);
%     hold on;
% end
% %plot(xaccfull(1:1000),yaccfull(1:1000),'b.','MarkerSize', 10, 'DisplayName', 'Full');
% 
% h = legend('show','location','SouthEast');
% title('EF recovered from factor space');
% scatter(xaccfullrecover,yaccfullrecover,'ko');
% title('EF in reconstructed space');
% 
% subplot(2,2,4);
% scatter(xaccrecover,yaccrecover,'go');
% title('EF in rotated factor space');


% hold on;


