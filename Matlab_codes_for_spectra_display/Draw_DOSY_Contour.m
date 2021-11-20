function Draw_DOSY_Contour(Spec_grid, DiffCoef, diff_v, ppm, ContourLevel, linewidth, range_ppm, range_diff);
if nargin < 8 || isempty(range_diff)
    range_diff = [diff_v(1),diff_v(end)];
end
if nargin < 7 || isempty(range_ppm)
    range_ppm = [ppm(1), ppm(end)];
end
if nargin < 6 || isempty(linewidth)
    linewidth = 1;
end
if nargin < 5 || isempty(ContourLevel)
    ContourLevel = 20;
end
if nargin < 4||isempty(ppm)
    Nf = size(Spec_grid,2);
    ppm = 1: Nf;
end
if nargin < 3||isempty(diff_v)
    Nd = size(Spec_grid,1);
    diff_v = linspace(min(DiffCoef),max(DiffCoef),Nd);
end
%%
proj_ppm = sum(Spec_grid, 1);
proj_diff = sum(Spec_grid, 2);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure,
ax1 = axes('position',[0.1 0.85 0.795 0.13]);
plot(ax1,ppm,proj_ppm/max(abs(proj_ppm(:))),'k-','LineWidth',1);
axis off;
set(gca,'Xdir','reverse');
xlim(range_ppm);ylim([-0.2,1]);
ax2 = axes('position',[0.1 0.20 0.795 0.65]);
contour(ax2,ppm,diff_v,Spec_grid,ContourLevel,'linewidth',linewidth)
xlabel(ax2,'Chemical Shift (ppm)')
ylabel(ax2,'Diffusion Coefficient (10^{-10} m^2/s)')
set(ax2,'Ydir','reverse','Xdir','reverse');
set(ax2,'YTick',unique(DiffCoef) );
xlim(range_ppm);
ylim(range_diff);
for i = 1:length(DiffCoef)
    line(ax2,get(ax2,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
end
ax3 = axes('position',[0.91 0.20 0.08 0.65]);
plot(ax3,proj_diff/max(abs(proj_diff)),diff_v,'k-','linewidth',1);
set(ax3,'Ydir','reverse');
ylim(range_diff); xlim([0 1]);
axis off;


end

