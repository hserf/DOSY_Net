%% Simulated data (applied in Supporting Information - Section A)
clc;
clear;
close all;
file_path = fileparts(mfilename('fullpath'));

%%
rng(1)
%%%%%%%%%%%%%%%%%%
n = 14;
% n = 20;
b = 0:n-1;
pn = 10;
ppm = linspace(0.05,0.95,pn);
IndFre = {[1,3,10],[2,5,7], [4,6,8,9]};

alpha = [0.1,0.2,0.3];
cn = length(alpha);
%%%%%%%%%%%%%%%%%%
% fn = length(ppm);
Aksave = zeros(pn,length(alpha));

for it = 1:cn
    DifC(:,it) = exp(-alpha(it)*b.' );
    ind_tmp = IndFre{it};
    Amp{it}= 0.8*rand(length(ind_tmp),1)+0.2;
    tmp = zeros(1,pn);
    tmp(ind_tmp) = Amp{it};
    FC(it,:) = tmp;
    Aksave(IndFre{it},it) = Amp{it};
end
figure,hold on,
plot(DifC)
DOSYExp = DifC * FC;

normedDOSYExp = max(DOSYExp(:));
DOSYExp = DOSYExp/normedDOSYExp;
sigma2 = 0.015;
% sigma2 = 0.0;
noise =  sigma2*randn(size(DOSYExp));
DOSYExp_noise = DOSYExp + noise;

S = DOSYExp_noise;
S = S.';
idx_peaks = 1:size(S,1);

Aksave = Aksave.';

% rec_S = Aksave*exp(-alpha(:)*b);
% Aksave = Aksave./repmat(rec_S(:,1),1,cn);
% rec_S = Aksave*exp(-alpha(:)*b);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nf = size(FC,2);
ff = 0:1/Nf:1-1/Nf;
sgm_d = 0.01; sgm_f = 0.01; diff_v = linspace(0,0.5,100); 
ContourLevel = 40;
Spec_grid = Par2Spectr_DOSY_nfreq(alpha,DifC,FC,1:Nf, sgm_d, sgm_f, diff_v, ff);
Draw_DOSY_Contour(Spec_grid, alpha, diff_v, ff, ContourLevel);

save([file_path,'\testdataSigma',...
    num2str(sigma2),'.mat'],'S','b','idx_peaks','ppm')
save([file_path,'\trueparaSigma',...
    num2str(sigma2),'.mat'],'alpha','Aksave')

return
