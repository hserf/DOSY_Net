function Spec_grid = Par2Spectr_DOSY_nfreq(DiffCoef,Decay,Spectrum, idx_freq, sgm_d, sgm_f, diff_v, ppm);
%% This function generate a whole DOSY spectrum from the DOSY parameter estimation result
%% Input: 
%% -DiffCoef: the estimated diffusion coefficients, is a vector or column containing n_decay elements
%% -Decay: the decay curves corresponding to the estimated diffusion coefficients **unused now**
%% -Spectrum: with size [n_decay, n_freq]
%% -idx_freq: the index of the frequency points in Spectrum 
%% -sgm_d: the peak linewidth of the diffusion coefficient dimension
%% -sgm_d: the peak linewidth of the frequency dimension
%% -diff_v: the grids on the diffusion coefficient dimension, with a length of Nd
%% -ppm: the grids on the frequency dimension, with a length of Nf
%% OUTPUT: 
%% -Spec_grid with size [Nd, Nf]
[n_decay, n_freq] = size(Spectrum);
if nargin < 8||isempty(ppm)
    Nf = 1e3;
    ppm = 1: Nf;
else
    Nf = length(ppm);
end
if nargin < 7||isempty(diff_v)
    Nd = 1e2;
    diff_v = linspace(min(DiffCoef),max(DiffCoef),Nd);
else
    Nd = length(diff_v);
end
if nargin < 6||isempty(sgm_f)
    sgm_f = 1;
end
if nargin < 5||isempty(sgm_d)
    sgm_d = (diff_v(2)-diff_v(1))*2;
end

if length(idx_freq) == length(ppm)
    spec_whole = Spectrum;
else
    spec_whole = zeros(n_decay, Nf);
    nf = 1:Nf;
    for k1 = 1:n_decay
        spec_whole(k1,:) = Spectrum(k1,:)*exp(-(idx_freq(:)-nf).^2/2/sgm_f^2);
    end
end

Spec_grid = zeros(Nd, Nf);
for k = 1: n_decay
    dk = DiffCoef(k);
    tmp = exp(-(diff_v(:)-dk).^2/2/sgm_d^2)*spec_whole(k,:);
    Spec_grid = Spec_grid+tmp;
end

end

