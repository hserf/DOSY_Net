%% showing the resulting spectrum for QGC
clear all; close all;
file_path = fileparts(mfilename('fullpath'));
addpath(file_path)

is_mat_file = 0;
file_path = '..\Net_Results\QGC\';
SubFolderNames = dir(file_path);
file_folder = strcat(file_path,SubFolderNames(end).name),% Find the latest folder
load(strcat(file_folder,'\data_org.mat'));
idx_peaks = double(idx_peaks);

beta0 = 0.1;
%% Read in the result
dc_z = csvread(strcat(file_folder, '\diffusion_coeffs.csv'));
Sp_z = csvread(strcat(file_folder, '\Sp.csv'));
[N_iter, N_d] = size(dc_z);
N_freq = round(size(Sp_z,2)/N_d);
Nf = length(ppm);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot the changing of diffusion coefficients through the iterations
N_iter = size(dc_z,1);
iter = (0:N_iter-1)*500;
figure, plot(iter, dc_z,'linewidth',1.5);
xlabel('Iterations'); ylabel('Diffusion Coefficient (10^{-10} m^2/s)')
%% 
fidelity_loss = zeros(1,N_iter);
sparsity = zeros(1,N_iter);
for k = 1: N_iter
    dc = dc_z(k,:);
    Sp = reshape(Sp_z(k,:),[N_d,N_freq]);
    Decay = exp(-b(:)*dc);
    X_rec = Decay*Sp;
    X_rec = X_rec.';
    fidelity_loss(k) = norm(X_rec(:)-S(:))^2;
    lambda_dc = sum(Decay.^2,1);
    Sp_norm = Sp./lambda_dc(:);
    Sp_norm = Sp_norm.*max(S,[],2).';
    sparsity(k) = sum(abs(Sp_norm(:)));
end
figure, subplot(211); semilogy(iter, fidelity_loss,'k','linewidth',1.5)
xlabel('Iterations'); ylabel('Fidelity Loss');
subplot(212); semilogy(iter, sparsity,'k','linewidth',1.5)
xlabel('Iterations'); ylabel('Sparsity');
total_loss = fidelity_loss+beta0*sparsity;
figure, semilogy(iter, total_loss,'k','linewidth',1.5);
xlabel('Iterations'); ylabel('Total Loss');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Final output 
k = N_iter;
dc = dc_z(k,:);
Sp = reshape(Sp_z(k,:),[N_d,N_freq]);
Decay = exp(-b(:)*dc);
X_rec = Decay*Sp;
%%%
sp0 = sum(Sp(:)); 
Sp_norm = Sp./lambda_dc(:);
sp1 = sum(Sp_norm(:));
Sp_norm = Sp.*max(S,[],2).';
sp2 = sum(Sp_norm(:));
Sp_norm = Sp.*sqrt(lambda_dc(:));
sp3 = sum(Sp_norm(:));

% [fidelity_loss(end),sparsity(end),sp0,sp1,sp2,sp3]
[fidelity_loss(end),sparsity(end),sp0,sp3]
%%%
%% Draw contour figure for the final output
ContourLevel = 40; linewidth = 1.5;
range_ppm = [4 12.5];
range_diff = [1, 12];
Dn = 100;
diff_v = linspace(range_diff(1),range_diff(2),Dn);
sgm_f = 1; % linewidth in frequency dimension
sgm_d = 0.1; % linewidth in diffusion coefficient dimension

dc_round = roundn(dc,-1);
Spec_grid = par2spectr_DOSY(dc_round,Sp,idx_peaks, sgm_d, sgm_f, diff_v, ppm); % generate a speudo DOSY spectrum

Spec_grid = Spec_grid/max(abs(Spec_grid(:)));
Draw_DOSY_Contour(Spec_grid, dc_round, diff_v, ppm, ContourLevel, linewidth, range_ppm, range_diff);

%% Plot the spectra of each diffusion components
[dc_s, ind] = sort(dc);
Sp_s = Sp(ind,:);
sgm_f = 3;
nf = 1:length(ppm);
figure,
for k = 1: length(dc)
    spec_k = Sp_s(k,:)*exp(-(idx_peaks(:)-nf).^2/2/sgm_f^2);
    subplot(length(dc),1,k); plot(ppm, spec_k, 'k','linewidth',1);
    set(gca,'xdir','reverse');
    xlim([range_ppm(1),range_ppm(2)]);
    text(mean(range_ppm),max(spec_k)/2, ['D_{',num2str(k),'} = ',num2str(roundn(dc_s(k),-2))]);
end
xlabel('ppm')

