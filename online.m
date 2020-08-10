function B_test = online(Wx1, Wx2, mux1, mux2, learning_rate, D, iter_num,...
    HashCode_length, X1_test, X2_test)

X1_test = X1_test';
X2_test = X2_test';
B_test = rand(HashCode_length, size(X1_test, 2));

for i = 1 : iter_num  
    B_temp = (1 / mux1) * Wx1 * X1_test + (1 / mux2) * Wx2 * X2_test;
    B_test = (1 - learning_rate) * B_test + learning_rate * B_temp;
    B_test = sign(B_test);
    B_test(B_test == 0) = -1;
    hx1 = norm(B_test - Wx1 * X1_test);
    hx2 = norm(B_test - Wx2 * X2_test);
    mux1 = hx1 / (hx1 + hx2);
    mux2 = hx2 / (hx1 + hx2);
end
end