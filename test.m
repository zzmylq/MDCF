HashCode_length = 8;
rank = 4;
alpha1 = 0.01;
alpha2 = 0.01;
beta = 0.01;
gamma = 0.0001;
lambda = 0.000001;
iter_num = 50;
learning_rate = 0.8;
factor = 0.9;
k = 20;
load('demo_data.mat');

X1_test = X1_test .* round(rand(1,size(X1_test,1)))';
X2_test = X2_test .* round(rand(1,size(X2_test,1)))';

X1 = X1';
X2 = X2';
Y1 = Y1';
Y2 = Y2';

[Wx1, Wx2, Wy1, Wy2, Hx, Hy, Rx, Ry, ZRx, ZRy, GRx, GRy, D, B]...
    = MDCFinit(U, Sigma, VT, X1, X2, Y1, Y2, HashCode_length, rank,...
    alpha1, alpha2, beta, gamma, lambda, iter_num);

[Wx1, Wx2, Wy1, Wy2, D, B, mux1, mux2, muy1, muy2]...
    = MDCF(Wx1, Wx2, Wy1, Wy2, Hx, Hy, Rx, Ry, ZRx, ZRy, GRx, GRy, D, B,...
    U, Sigma, VT, X1, X2, Y1, Y2, HashCode_length, rank,...
    alpha1, alpha2, beta, gamma, lambda, iter_num, learning_rate, factor);

B_test = online(Wx1, Wx2, mux1, mux2, learning_rate, D, iter_num,...
    HashCode_length, X1_test, X2_test);

[ndcg,hit,~] = rating_metric_hit(test, B_test', D', k, neg, CS_id);
fprintf('ndcg:\n');
fprintf('%0.4f\n',ndcg);
fprintf('hit:\n');
fprintf('%0.4f\n',hit);
