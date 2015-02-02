% Run BTL SDP and create a plot.
%  - Fix d.
%  - Vary n.
function [] = BTLSDP_test1(outdir)

if ~exist('outdir', 'var')
    outdir = '.';
end

wdists = {'uniform', 'sqrtlinear', 'linear'};

d = 10;
sigma = .3;
a = 1;
b = 1;
ntrials = 5;

max_n = 10 * d * log(d);
num_n = 10;
n_vals = ceil(exp(log(d):((log(max_n)-log(d))/num_n):log(max_n)));

w_est_diff_L2normed_means = cell(length(wdists),1);
w_est_diff_L2normed_stderrs = cell(length(wdists),1);
w_obj_ratio_means = cell(length(wdists),1);
w_obj_ratio_stderrs = cell(length(wdists),1);

max_runtime = 0;
avg_runtime = 0;
for wdist_i = 1:length(wdists)
    wdist = wdists{wdist_i};
    outfilename = sprintf('BTLSDP_test1.%s.d%d.sigma%g.a%g.b%g.out', wdist, d, sigma, a, b);
    fid = fopen([outdir '/' outfilename], 'w');
    fprintf(fid, 'BTLSDP_test1\n');
    fprintf(fid, 'wdist: %s\n', wdist);
    fprintf(fid, 'd: %d\n', d);
    fprintf(fid, 'sigma: %g\n', sigma);
    fprintf(fid, '(a, b): (%g, %g)\n', a, b);
    fprintf(fid, 'ntrials: %d\n', ntrials);
    fprintf(fid, '\n');
    fprintf(fid, ['n\tw_loss\tw_obj\tw_est_loss\tw_est_obj' ...
                  '\tw_est0_loss\tw_est0_obj\tw_L2norm' ...
                  '\tw_est_L2norm\tw_est_diff_L2normed' ...
                  '\tw_est0_L2norm\tw_est0_diff_L2normed' ...
                  '\teigengap_W_normed\truntime\n']);
    w_est_diff_L2normed_ = zeros(ntrials,length(n_vals));
    w_obj_ratio_ = zeros(ntrials,length(n_vals));
    for i = 1:length(n_vals)
        for trial = 1:ntrials
            n = n_vals(i);
            [w_loss, w_obj, w_est_loss, w_est_obj, w_est0_loss, w_est0_obj, ...
             w_L2norm, w_est_L2norm, w_est_diff_L2normed, w_est0_L2norm, w_est0_diff_L2normed, ...
             eigengap_normed, runtime] ...
                = BTLSDP(wdist, d, n, sigma, a, b);
            fprintf(fid, '%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n', ...
                    n, w_loss, w_obj, w_est_loss, w_est_obj, w_est0_loss, w_est0_obj, ...
                    w_L2norm, w_est_L2norm, w_est_diff_L2normed, w_est0_L2norm, w_est0_diff_L2normed, ...
                    eigengap_normed, runtime);
            w_est_diff_L2normed_(trial,i) = w_est_diff_L2normed;
            w_obj_ratio_(trial,i) = w_est_obj / w_obj;
            max_runtime = max(max_runtime, runtime);
            avg_runtime = avg_runtime + runtime;
        end
    end
    fclose(fid);
    
    w_est_diff_L2normed_means{wdist_i} = mean(w_est_diff_L2normed_);
    w_est_diff_L2normed_stderrs{wdist_i} = std(w_est_diff_L2normed_) ./ sqrt(ntrials);
    w_obj_ratio_means{wdist_i} = mean(w_obj_ratio_);
    w_obj_ratio_stderrs{wdist_i} = std(w_obj_ratio_) ./ sqrt(ntrials);
end
avg_runtime = avg_runtime / (length(wdists) * length(n_vals) * ntrials);

fprintf('RUNTIMES: (avg, max) = (%g, %g)\n', avg_runtime, max_runtime);

params_title = sprintf('d=%d, sigma=%g, (a,b)=(%g,%g), %d trials', d,sigma,a,b,ntrials);

plot_param_string = sprintf('d%d.sigma%g.a%g.b%g', d, sigma, a, b);

figure;
for wdist_i = 1:length(wdists)
    errorbar(n_vals ./ d, w_est_diff_L2normed_means{wdist_i}, w_est_diff_L2normed_stderrs{wdist_i});
    if wdist_i == 1
        hold all;
    end
end
legend(wdists);
xlabel('n/d');
ylabel('||what-w*||/||w*||');
title({'Test1: w Error'; params_title});
saveas(gcf, [outdir '/BTLSDP_test1.' plot_param_string '.wdiff.pdf']);

clf;
for wdist_i = 1:length(wdists)
    errorbar(n_vals ./ d, w_obj_ratio_means{wdist_i}, w_obj_ratio_stderrs{wdist_i});
    if wdist_i == 1
        hold all;
    end
end
legend(wdists);
xlabel('n/d');
ylabel('Obj(what)/Obj(w*)');
title({'Test1: Objective Ratio'; params_title});
saveas(gcf, [outdir '/BTLSDP_test1.' plot_param_string '.objratio.pdf']);

end
