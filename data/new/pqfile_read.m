clear all; close all;
load('D:\my_work\mypaper\dosy_param\1st_revision\codes\data\new\pqfile.mat');
g = [0.0006,0.0294,0.0581,0.0868,0.1156,0.1443,0.1730,0.2017,0.2305,0.2592,0.2880,0.3166,0.3454,0.3741,0.4029]*1e2;
BD = 0.2; % diffusion time
LD = 0.002; % diffusion encoding time
gamma = 4257.7;
g2 = (2*pi*gamma*g*LD).^2*(BD-LD/3)*1e4;
b = g2*1e-10;

figure,plot(result2ddata(:,1))
figure,plot(result2ddata(1,:));set(gca,'xdir','reverse')
figure,plot(sum(result2ddata));set(gca,'xdir','reverse')

S = result2ddata/max(result2ddata(:));
S0 = S(1,:);
idx = find(S0>0.01);
figure,plot(S0);
hold on; plot(idx,S0(idx),'r.')
S = S.';
save('D:\my_work\mypaper\dosy_param\1st_revision\codes\data\new\pqfile_net_input.mat','S','b');
%%%%%%%%%%%

