clear all; close all;

load('..\data\simulation\testdataSigma0.1.mat');
load('..\data\simulation\trueparaSigma0.1.mat')
file_path = '..\Net_Results\sim\';
SubFolderNames = dir(file_path);
file_folder = [file_path,SubFolderNames(end).name], % Find the latest folder

a_z = csvread([file_folder, '\diffusion_coeffs.csv']);
A_z = csvread([file_folder, '\Sp.csv']);
[N_iter, N_d] = size(a_z);
N_freq = size(A_z,2)/N_d;
k = N_iter;
ak = a_z(k,:);
Ak = reshape(A_z(k,:),[N_d,N_freq]);

Decay = exp(-b(:)*ak);
X_rec = Decay*Ak;
%%
Nf = size(S,1);
ff = 0:1/Nf:1-1/Nf;
sgm_d = 0.005; sgm_f = 0.001; diff_v = linspace(0,0.5,100); 
ContourLevel = 20;
Spec_grid = par2spectr_DOSY(alpha,Aksave,1:Nf, sgm_d, sgm_f, diff_v, ff);
Draw_DOSY_Contour(Spec_grid, alpha, diff_v, ff, ContourLevel);

Spec_grid_rec = par2spectr_DOSY(ak,Ak,1:size(Ak,2), sgm_d, sgm_f, diff_v, ff);
Draw_DOSY_Contour(Spec_grid_rec, ak, diff_v, ff, ContourLevel);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% display the convergence curve
N_iter = size(a_z,1);
iter = 1:N_iter;
fidelity_loss = zeros(1,N_iter);
for k = 1: N_iter
    dc = a_z(k,:);
    Sp = reshape(A_z(k,:),[N_d,N_freq]);
    Decay = exp(-b(:)*dc);
    X_rec = Decay*Sp;
    X_rec = X_rec.';
    fidelity_loss(k) = norm(X_rec(:)-S(:))^2;
    lambda_dc = sum(Decay.^2,1);
end
figure, semilogy(iter, fidelity_loss,'k','linewidth',1.5)
xlabel('Iterations'); ylabel('Fidelity Loss');
% return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
color_list = {'k','r','b'};
figure, subplot(131);hold on;
for k = 1:size(Aksave,1)
    idx = find(Aksave(k,:));
    xn_k = S(idx,:);
    plot(b,exp(-alpha(k)*b),'g:','linewidth',5);
    plot(b,xn_k,color_list{k});
end
xlim([0,b(end)]);
ylim([-0.05,1])
%%
model_fun = @(c,b) c(1)*exp(-c(2)*b);
xn_fit = zeros(size(S));
Ak_expfit = zeros(1,size(S,1));
ak_expfit = zeros(1,size(S,1));
subplot(132); hold on;
for k = 1:size(S,1)
    xn_k = S(k,:);
    mdl = fitnlm(b,xn_k,model_fun,[1,0.1]);
    coefficients = mdl.Coefficients{:, 'Estimate'};
    A_k = coefficients(1); Ak_expfit(k) = A_k;
    a_k = coefficients(2); ak_expfit(k) = a_k;
    xn_fit(k,:) = A_k*exp(-a_k*b);
    plot(b, xn_fit(k,:),'r')
end
xlim([0,b(end)]);
ylim([-0.05,1])

subplot(133), hold on;
for k = 1:length(ak)
    xn_net = exp(-ak(k)*b);
    plot(b, xn_net,'k')
end
xlim([0,b(end)]);
ylim([-0.05,1])
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
%% Display monofitting spectrum
FC_expfit = diag(Ak_expfit);
Spec_grid_expfit = par2spectr_DOSY(ak_expfit,FC_expfit,1:size(S,1), sgm_d, sgm_f, diff_v,ff);
Draw_DOSY_Contour(Spec_grid_expfit, ak_expfit, diff_v, ff, ContourLevel);

