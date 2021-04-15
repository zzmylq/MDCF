function [ndcg,hit,rmse] = rating_metric_hit3(test, P, Q, k, neg, CS_id)
[I,J,V] = find(test);
pred_val = sum(P(I,:) .* Q(J,:), 2);
all_col = [I,J,V,pred_val];
rmse = sqrt(mean((V - pred_val).^2));
act_col = sortrows(all_col, [1,-3]);
pred_col = sortrows(all_col,[1,-4]);
user_count = full(sum(test>0,2));
cum_user_count = cumsum(user_count);
cum_user_count = [0;cum_user_count];
num_users = size(test,1);
uind = 1;
ndcg_all = zeros(num_users - sum(sum(test>0)==0),k);
neg_num = 999;

for u=1:num_users 
    if user_count(u) == 0
        continue;
    end
    u_start = cum_user_count(u)+1;
    u_end = cum_user_count(u+1);
    act = act_col(u_start:u_end,3);
    discount = log2((1:k)'+1);
    pred = pred_col(u_start:u_end,3);
    if k > length(act)
        act_extend = [act; zeros(k-length(act),1)];
        pred_extend = [pred; zeros(k-length(act),1)];
    else
        act_extend = act(1:k);
        pred_extend = pred(1:k);
    end
    idcg = cumsum((2.^act_extend - 1) ./ discount);
    dcg = cumsum((2.^pred_extend - 1) ./discount);
    ndcg_all(uind,:) = dcg ./ idcg;
    uind = uind + 1;
end
ndcg = mean(ndcg_all);

hit_all = zeros(num_users - sum(sum(test>0)==0),k);

CS_col = 1:size(CS_id,1);
CS_col = CS_col';
CSid_extend = repmat(CS_col', neg_num, 1)';
CSid_extendT = CSid_extend';
CSid_extend_list = CSid_extendT(:);
neg_lookup = neg(CS_id+1,1:neg_num);
neg_lookupT = neg_lookup';
neg_list = neg_lookupT(:) + 1;
values_list = zeros(size(neg_list,1),1);


pred_val2 = sum(P(CSid_extend_list,:) .* Q(neg_list,:), 2);
all_col2 = [CSid_extend_list, neg_list, values_list, pred_val2];
act_col2 = sortrows(all_col2, [1,-4]);

act_ratings = act_col2(:,4);
for ki = 1:k
    k_ratings = act_ratings(ki:neg_num:size(act_ratings,1));
    for i = 1:size(CS_id,1)
        target_col = 1; 
        target_val = i; 
        [row,~] = find(act_col(:,target_col)==target_val); 
        
        zong = act_ratings((i-1)*neg_num+1 : (i*neg_num));
        zong_count = sum(zong == k_ratings(i));
        topk = act_ratings((i-1)*neg_num+1 : (i-1)*neg_num+ki);
        topk_count = sum(topk == k_ratings(i));
        
        hit_num = sum(act_col(row,4)>k_ratings(i));
        hit_num = hit_num + sum(act_col(row,4)==k_ratings(i))*(topk_count/zong_count);
        
        %hit_num = sum(act_col(row,4)>=k_ratings(i));
        hit_all(i,ki) = hit_num / size(row,1);
    end
end


hit = mean(hit_all);
end
