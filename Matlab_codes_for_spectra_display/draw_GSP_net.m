%% showing the resulting spectrum for GSP
clear all; close all;
file_path = fileparts(mfilename('fullpath'));
addpath(file_path)

load('..\data\GSP\GSP_net_input.mat');
Nf = length(ppm);

file_path = '..\Net_Results\GSP\';
SubFolderNames = dir(file_path);
file_folder = [file_path,SubFolderNames(end-1).name],% Find the latest folder

%% Read in the result
dc_z = csvread([file_folder, '\diffusion_coeffs.csv']);
Sp_z = csvread([file_folder, '\Sp.csv']);
[N_iter, N_d] = size(dc_z);
k = N_iter;
dc = dc_z(k,:);
N_freq = round(size(Sp_z,2)/N_d);
Sp = reshape(Sp_z(k,:),[N_d,N_freq]);
Decay = exp(-b(:)*dc);
X_rec = Decay*Sp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% final loss
norm(X_rec.'-S,'fro')^2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Draw contour figure
ContourLevel = 20;
linewidth = 1.5;
range_ppm = [3, 5.5];
range_diff = [1.5, 5];
Dn = 100;
diff_v = linspace(range_diff(1),range_diff(2),Dn);
sgm_f = 1; % linewidth in frequency dimension
sgm_d = 0.05; % linewidth in diffusion coefficient dimension

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
    spec_k = Sp_s(k,:)*exp(-(idx_peaks'-nf).^2/2/sgm_f^2);
    subplot(length(dc),1,k); plot(ppm, spec_k, 'k','linewidth',1);
    set(gca,'xdir','reverse');
    xlim([range_ppm(1),range_ppm(2)]);
    text(mean(range_ppm),max(spec_k)/2, ['D_{',num2str(k),'} = ',num2str(roundn(dc_s(k),-2))]);
end
xlabel('ppm')

