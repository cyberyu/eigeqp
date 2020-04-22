function xopt = optimal_sol(Q,A,b,c,lambda)
    %invQ = Q\eye(size(Q));
    invQ = pinv(Q);
    disp('job done');
    xopt = -lambda*(invQ-invQ*A'*inv(A*invQ*A')*A*invQ)*c + invQ*A'*inv(A*invQ*A')*b;
end