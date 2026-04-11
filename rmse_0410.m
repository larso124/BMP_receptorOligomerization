clear all
type = 'both';

load(sprintf("data/MI_%s.mat",type))

% exclude first hour
% blue = MI_t_all(1,61:end);
% green = MI_t_all(2,61:end);
% red = MI_t_all(3,61:end);

% all data
blue = MI_t_all(1,:);
green = MI_t_all(2,:);
red = MI_t_all(3,:);

max(green)
x = max(green)*ones(1,179);
% x = x(:,61:end);
% grmse = immse(x, green);
% rrmse = immse(x, red);
% brmse = immse(x, blue);

gr = rmse(x,green)
rr = rmse(x,red)
br = rmse(x,blue)

% bvg = rmse(blue, green);
% bvr = rmse(blue, red);
% rvg = rmse(red, green);