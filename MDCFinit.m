function [Wx1, Wx2, Wy1, Wy2, Hx, Hy, Rx, Ry, ZRx, ZRy, GRx, GRy, D, B]...
    = MDCFinit(U, Sigma, VT, X1, X2, Y1, Y2, HashCode_length, rank,...
    alpha1, alpha2, beta, gamma, lambda, iter_num, learning_rate)

Wx1 = rand(HashCode_length, size(X1,1));
Wx2 = rand(HashCode_length, size(X2,1));
Wy1 = rand(HashCode_length, size(Y1,1));
Wy2 = rand(HashCode_length, size(Y2,1));
Rx = rand(HashCode_length, HashCode_length);
Ry = rand(HashCode_length, HashCode_length);
Hx = rand(HashCode_length, size(U,1));
Hy = rand(HashCode_length, size(VT,2));
ZRx = rand(HashCode_length, HashCode_length);
ZRy = rand(HashCode_length, HashCode_length);
GRx = rand(HashCode_length, HashCode_length);
GRy = rand(HashCode_length, HashCode_length);
B = rand(HashCode_length, size(U,1))*2-1;
D = rand(HashCode_length, size(VT,2))*2-1;

[Vx1, ~, ~] = svd(Wx1 * Wx1');
Vx1 = Vx1(:,rank + 1 : HashCode_length);
[Vx2, ~, ~] = svd(Wx2 * Wx2');
Vx2 = Vx2(:,rank + 1 : HashCode_length);
[Vy1, ~, ~] = svd(Wy1 * Wy1');
Vy1 = Vy1(:,rank + 1 : HashCode_length);
[Vy2, ~, ~] = svd(Wy2 * Wy2');
Vy2 = Vy2(:,rank + 1 : HashCode_length);

hx1 = norm((Hx - Wx1 * X1),2);
hx2 = norm((Hx - Wx2 * X2),2);
mux1 = hx1 / (hx1 + hx2);
mux2 = hx2 / (hx1 + hx2);
hy1 = norm((Hy - Wy1 * Y1),2);
hy2 = norm((Hy - Wy2 * Y2),2);
muy1 = hy1 / (hy1 + hy2);
muy2 = hy2 / (hy1 + hy2);
    
U = U';
VT = VT';

for i = 1 : iter_num  
    
    hx1 = norm((Hx - Wx1 * X1),2);
    hx2 = norm((Hx - Wx2 * X2),2);
    mux1_update = hx1 / (hx1 + hx2);
    mux2_update = hx2 / (hx1 + hx2);
    hy1 = norm((Hy - Wy1 * Y1),2);
    hy2 = norm((Hy - Wy2 * Y2),2);
    muy1_update = hy1 / (hy1 + hy2);
    muy2_update = hy2 / (hy1 + hy2);
    
    Wx1_update = W_update(gamma, mux1, Vx1, X1, Hx);
    Wx2_update = W_update(gamma, mux2, Vx2, X2, Hx);
    Wy1_update = W_update(gamma, muy1, Vy1, Y1, Hy);
    Wy2_update = W_update(gamma, muy2, Vy2, Y2, Hy);
    
    Cx = 2 * Ry * Hy * VT * Sigma * U * Hx'...
        - Ry * Hy * Hy' * Ry' * ZRx * Hx * Hx'...
        + 2 * alpha1 * B * Hx' + lambda * ZRx - GRx;
    [CPx, ~, CQx] = svd(Cx);
    Rx_update = CPx * CQx;
    Cy = 2 * Rx * Hx * U' * Sigma * VT' * Hy'...
        - Rx * Hx * Hx' * Rx' * ZRy * Hy * Hy'...
        + 2 * alpha2 * D * Hy' + lambda * ZRy - GRy;
    [CPy, ~, CQy] = svd(Cy);
    Ry_update = CPy * CQy;
    
    Hx_p1 = Rx' * Ry * Hy * Hy' * Ry' * Rx...
            + (alpha1 + beta * (1 / mux1 + 1 / mux2)) * eye(HashCode_length);
    Hx_p2 = Rx' * Ry * Hy * VT * Sigma * U...
            + alpha1 * Rx' * B + (beta / mux1) * Wx1 * X1 + (beta / mux2) * Wx2 * X2;
    Hx_update = inv(Hx_p1) * Hx_p2;
    Hy_p1 = Ry' * Rx * Hx * Hx' * Rx' * Ry...
            + (alpha2 + beta * (1 / muy1 + 1 / muy2)) * eye(HashCode_length);
    Hy_p2 = Ry' * Rx * Hx * U' * Sigma * VT'...
            + alpha2 * Ry' * D + (beta / muy1) * Wy1 * Y1 + (beta / muy2) * Wy2 * Y2;
    Hy_update = inv(Hy_p1) * Hy_p2;
    
    D_update = Ry * Hy;
    B_update = Rx * Hx;
    
    [Vx1P, ~, ~] = svd(Wx1 * Wx1');
    Vx1_update = Vx1P(: ,rank + 1 : HashCode_length);
    [Vx2P, ~, ~] = svd(Wx2 * Wx2');
    Vx2_update = Vx2P(: ,rank + 1 : HashCode_length);
    [Vy1P, ~, ~] = svd(Wy1 * Wy1');
    Vy1_update = Vy1P(: ,rank + 1 : HashCode_length);
    [Vy2P, ~, ~] = svd(Wy2 * Wy2');
    Vy2_update = Vy2P(: ,rank + 1 : HashCode_length);
    
    CZRx = lambda * Rx - Ry * Hy * Hy' * Ry' * Rx * Hx * Hx';
    [CZRxP, ~, CZRxQ] = svd(CZRx);
    ZRx_update = CZRxP * CZRxQ;
    CZRy = lambda * Ry - Rx * Hx * Hx' * Rx' * Ry * Hy * Hy';
    [CZRyP, ~, CZRyQ] = svd(CZRy);
    ZRy_update = CZRyP * CZRyQ;
    
    GRx_update = GRx + lambda * (Rx - ZRx);
    GRy_update = GRy + lambda * (Ry - ZRy);
    
    mux1 = (1 - learning_rate) * mux1 + learning_rate * mux1_update;
    mux2 = (1 - learning_rate) * mux2 + learning_rate * mux2_update;
    muy1 = (1 - learning_rate) * muy1 + learning_rate * muy1_update;
    muy2 = (1 - learning_rate) * muy2 + learning_rate * muy2_update;
    Wx1 = (1 - learning_rate) * Wx1 + learning_rate * Wx1_update;
    Wx2 = (1 - learning_rate) * Wx2 + learning_rate * Wx2_update;
    Wy1 = (1 - learning_rate) * Wy1 + learning_rate * Wy1_update;
    Wy2 = (1 - learning_rate) * Wy2 + learning_rate * Wy2_update;
    Rx = (1 - learning_rate) * Rx + learning_rate * Rx_update;
    Ry = (1 - learning_rate) * Ry + learning_rate * Ry_update;
    Hx = (1 - learning_rate) * Hx + learning_rate * Hx_update;
    Hy = (1 - learning_rate) * Hy + learning_rate * Hy_update;
    D = (1 - learning_rate) * D + learning_rate * D_update;
    B = (1 - learning_rate) * B + learning_rate * B_update;
    
    Vx1 = (1 - learning_rate) * Vx1 + learning_rate * Vx1_update;
    Vx2 = (1 - learning_rate) * Vx2 + learning_rate * Vx2_update;
    Vy1 = (1 - learning_rate) * Vy1 + learning_rate * Vy1_update;
    Vy2 = (1 - learning_rate) * Vy2 + learning_rate * Vy2_update;
    ZRx = (1 - learning_rate) * ZRx + learning_rate * ZRx_update;
    ZRy = (1 - learning_rate) * ZRy + learning_rate * ZRy_update;
    GRx = (1 - learning_rate) * GRx + learning_rate * GRx_update;
    GRy = (1 - learning_rate) * GRy + learning_rate * GRy_update;
    
end

end

function W = W_update(gamma, mu, V, X, H)
    WA = gamma * V * V';
    WB = (1 / mu) * X * X';
    WC = (1 / mu) * H * X';
    W = sylvester(WA, WB, WC);
end
