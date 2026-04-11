clear all

% type = 'up';
% type = 'down';
type = 'both';

load(sprintf("data/MI_%s.mat",type))

% blue = MI_t_all(1,:);
% green = MI_t_all(2,:);
% red = MI_t_all(3,:);

blue = MI_t_all(1,121:end);
green = MI_t_all(2,121:end);
red = MI_t_all(3,121:end);

gaoc = trapz(green)
baoc = trapz(blue)
raoc = trapz(red)