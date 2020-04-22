% Q is K+I/lambda
% A is Y
% -lambda*c is vector 1,  c = ones(1,n),  lambda = -1
% seems in LSSVM formulation, b is 0

function xopt = optimal_sol_lssvm(Q,A,onevec)
    invQ = inv(Q);
    %xopt = (invQ-invQ*A'*inv(A*invQ*A')*A*invQ)*c + invQ*A'*inv(A*invQ*A')*b;
    xopt = (invQ-invQ*A'*inv(A*invQ*A')*A*invQ)*onevec; 
end

