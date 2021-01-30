HashCode_length = 128;
rank = 4;
alpha1 = 0.000001;
alpha2 = 10;
beta = 1;
gamma = 0.1;
lambda = 1;
iter_num = 20;
learning_rate = 0.02;
k = 20;
load('data.mat');

X1 = X1';
X2 = X2';
Y1 = Y1';
Y2 = Y2';

[Wx1, Wx2, Wy1, Wy2, Hx, Hy, Rx, Ry, ZRx, ZRy, GRx, GRy, D, B]...
    = MDCFinit(U, Sigma, VT, X1, X2, Y1, Y2, HashCode_length, rank,...
    alpha1, alpha2, beta, gamma, lambda, iter_num, learning_rate);

[Wx1, Wx2, Wy1, Wy2, D, B, mux1, mux2, muy1, muy2]...
    = MDCF(Wx1, Wx2, Wy1, Wy2, Hx, Hy, Rx, Ry, ZRx, ZRy, GRx, GRy, D, B,...
    U, Sigma, VT, X1, X2, Y1, Y2, HashCode_length, rank,...
    alpha1, alpha2, beta, gamma, lambda, iter_num, learning_rate);

B_test = online(Wx1, Wx2, mux1, mux2, learning_rate, D, iter_num,...
    HashCode_length, X1_test, X2_test);

ndcg = rating_metric(test, B_test', D', k);
fprintf('ndcg:%0.4f\n',ndcg);

