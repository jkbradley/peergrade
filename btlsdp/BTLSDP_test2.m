% Run BTL SDP and create a plot.
%  - Vary d
%  - Set n = d log(d) * sigma^2
function [] = BTLSDP_test2(outdir)

if ~exist('outdir', 'var')
    outdir = '.';
end

n_extra_scaling = 'const';  % const, d
n_multiplier = 3;

wdists = {'uniform', 'sqrtlinear', 'linear'};

d_vals = [5, 10, 20, 30];
sigma = .3;
a = 1;
b = 1;
ntrials = 5;

if strcmp(n_extra_scaling, 'const')
    n_vals = ceil((n_multiplier * square(sigma)) .* d_vals .* log(d_vals));
elseif strcmp(n_extra_scaling, 'd')
    n_vals = ceil((n_multiplier * square(sigma)) .* square(d_vals) .* log(d_vals));
else
    error(['Bad n_extra_scaling parameter: ' n_extra_scaling]);
end

w_est_diff_L2normed_means = cell(length(wdists),1);
w_est_diff_L2normed_stderrs = cell(length(wdists),1);
w_obj_ratio_means = cell(length(wdists),1);
w_obj_ratio_stderrs = cell(length(wdists),1);

max_runtime = 0;
avg_runtime = 0;
for wdist_i = 1:length(wdists)
    wdist = wdists{wdist_i};
    outfilename = sprintf('BTLSDP_test2.%s.sigma%g.a%g.b%g.N%s_%d.out', wdist, sigma, a, b, n_extra_scaling, n_multiplier);
    fid = fopen([outdir '/' outfilename], 'w');
    fprintf(fid, 'BTLSDP_test2\n');
    fprintf(fid, 'wdist: %s\n', wdist);
    fprintf(fid, 'sigma: %g\n', sigma);
    fprintf(fid, '(a, b): (%g, %g)\n', a, b);
    fprintf(fid, 'n_extra_scaling: %s\n', n_extra_scaling);
    fprintf(fid, 'n_multiplier: %d\n', n_multiplier);
    fprintf(fid, 'ntrials: %d\n', ntrials);
    fprintf(fid, '\n');
    fprintf(fid, ['d\tn\tw_loss\tw_obj\tw_est_loss\tw_est_obj' ...
                  '\tw_est0_loss\tw_est0_obj\tw_L2norm' ...
                  '\tw_est_L2norm\tw_est_diff_L2normed' ...
                  '\tw_est0_L2norm\tw_est0_diff_L2normed' ...
                  '\teigengap_W_normed\truntime\n']);
    w_est_diff_L2normed_ = zeros(ntrials,length(d_vals));
    w_obj_ratio_ = zeros(ntrials,length(d_vals));
    for i = 1:length(d_vals)
        for trial = 1:ntrials
            d = d_vals(i);
            n = n_vals(i);
            [w_loss, w_obj, w_est_loss, w_est_obj, w_est0_loss, w_est0_obj, ...
             w_L2norm, w_est_L2norm, w_est_diff_L2normed, w_est0_L2norm, w_est0_diff_L2normed, ...
             eigengap_normed, runtime] ...
                = BTLSDP(wdist, d, n, sigma, a, b);
            fprintf(fid, '%d\t%d\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n', ...
                    d, n, w_loss, w_obj, w_est_loss, w_est_obj, w_est0_loss, w_est0_obj, ...
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
avg_runtime = avg_runtime / (length(wdists) * length(d_vals) * ntrials);

fprintf('RUNTIMES: (avg, max) = (%g, %g)\n', avg_runtime, max_runtime);

if strcmp(n_extra_scaling, 'const')
    n_scaling_title = sprintf('n scaling: %d d log(d)*sigma^2', n_multiplier);
elseif strcmp(n_extra_scaling, 'd')
    n_scaling_title = sprintf('n scaling: %d d^2 log(d)*sigma^2', n_multiplier);
else
    error(['Bad n_extra_scaling parameter: ' n_extra_scaling]);
end
params_title = sprintf('sigma=%g, (a,b)=(%g,%g), %d trials', sigma,a,b,ntrials);

plot_param_string = sprintf('sigma%g.a%g.b%g.N%s_%d', sigma, a, b, n_extra_scaling, n_multiplier);

figure;
for wdist_i = 1:length(wdists)
    errorbar(d_vals, w_est_diff_L2normed_means{wdist_i}, w_est_diff_L2normed_stderrs{wdist_i});
    if wdist_i == 1
        hold all;
    end
end
legend(wdists);
xlabel('d');
ylabel('||what-w*||/||w*||');
title({'Test2: w Error'; params_title; n_scaling_title});
saveas(gcf, [outdir '/BTLSDP_test2.' plot_param_string '.wdiff.pdf']);

clf;
for wdist_i = 1:length(wdists)
    errorbar(d_vals, w_obj_ratio_means{wdist_i}, w_obj_ratio_stderrs{wdist_i});
    if wdist_i == 1
        hold all;
    end
end
legend(wdists);
xlabel('d');
ylabel('Obj(what)/Obj(w*)');
title({'Test2: Objective Ratio'; params_title; n_scaling_title});
saveas(gcf, [outdir '/BTLSDP_test2.' plot_param_string '.objratio.pdf']);

end
