clear all;
close all;

%how much time should you shift back (in seconds)
time_back_orig=2;
time_forward_orig=6;

%how many shuffled iterations to make distribution of F-stats
num_shuff=500;

%load gcamp and behavioral data of example recording
load('example_neuron.mat')

%convert seconds to hertz for relevant behavioral timestamps
NPTimes=output.NosePokeEnter.*g_output.samp_rate;
LeverPresent=output.LeverPresentation.*g_output.samp_rate;
LeverTimesI=output.LeverTimes(output.IpsPress==1).*g_output.samp_rate;
LeverTimesC=output.LeverTimes(output.IpsPress==-1).*g_output.samp_rate;
CSRew=output.RewardPresentation(output.RewardEnter~=0).*g_output.samp_rate;
CSNoRew=output.RewardPresentation(output.RewardEnter==0).*g_output.samp_rate;
RewardEnter=output.RewardEnter(output.RewardEnter~=0).*g_output.samp_rate;

%Define behaviors to be tested in model
cons={'NPTimes','LeverPresent','LeverTimesI','LeverTimesC'...
    'CSRew','CSNoRew','RewardEnter'};

%define time shift for each tested event, 0 for -2-6 seconds, 1 for 0-8 seconds
con_shift=[0 1 0 0 1 1 0];

%Z-score gcamp signal
gcamp_y=(g_output.gcamp-mean(g_output.gcamp))./std(g_output.gcamp);


%% Regression data prep

%Initialize x-matrices
con_iden=[];
x_basic=[];    %No interaction terms, simply event times
event_times_mat=[];
num_bins=numel(gcamp_y);

for con=1:numel(cons)

    if con_shift(con)==1
        time_back=0;
        time_forward=time_back_orig+time_forward_orig;
    else
        time_back=time_back_orig;
        time_forward=time_forward_orig;
    end

    %gets matrix of event times (in hertz)
    con_times=eval(cons{con});

    %Gets rid of abandonded trials
    con_times(con_times==0)=[];

    %Creates vector with binary indication of events
    con_binned=zeros(1,num_bins);
    con_binned(int32(con_times))=1;

    con_binned=circshift(con_binned,[0,-time_back*g_output.samp_rate]);
    event_times_mat=vertcat(event_times_mat,con_binned);

    %preloads basis set
    load('basis_81x25.mat')

    %convolves time series of events with basis sets
    for num_sets=1:numel(basis_set(1,:))
        temp_conv_vec=conv(con_binned,basis_set(:,num_sets));
        x_basic=horzcat(x_basic,temp_conv_vec(1:numel(con_binned))');
    end
    con_iden=[con_iden ones(1,size(basis_set,2))*con];

end

%mean center predictor matrix
x_all=mean_center(x_basic);

%makes predictor cell, identifier of groups for nested regressions
num_cons=unique(con_iden);
for i=1:numel(num_cons)
    preds_cells{i}=find(con_iden==i);
end

%% First, unshuffled F-stat
X = x_all;
y = gcamp_y';
preds_to_test_cell = preds_cells;

X = [ones(size(X,1),1) X];
n = size(X,1);  % number of observations
k = size(X,2);             % number of variables (including constant)

b = X \ y;                 % estimate b with least squares
u = y - X * b;             % calculates residuals
s2 = u' * u / (n - k);     % estimate variance of error term (assuming homoskedasticity, independent observations)
BCOV = inv(X'*X) * s2;     % get covariance matrix of b assuming homoskedasticity of error term etc...
bse = diag(BCOV).^.5;      % standard errors

clear Fp_vec
for l=1:length(preds_to_test_cell)
    preds_to_test_cell{l} = preds_to_test_cell{l}+1; %because of the constant

    R = zeros(length(preds_to_test_cell{l}),k);
    for l2 = 1:length(preds_to_test_cell{l})
        R(l2, preds_to_test_cell{l}(l2))=1;
    end

    r = zeros(length(preds_to_test_cell{l}),1);          % Testing restriction: R * b = r

    num_restrictions = size(R, 1);
    F = (R*b - r)'*inv(R * BCOV * R')*(R*b - r) / num_restrictions;   % F-stat (see Hiyashi for reference)
    F_vec(l) = F;
    Fp_vec(l) = 1 - fcdf(F, num_restrictions, n - k);  % F p-val
end

Fp_nonshuff(:)=Fp_vec;
F_nonshuff(:)=F_vec;
b_final(:)=b;

%% Next, shuffled F-stats
for i=1:num_shuff
    X = x_all;
    y=circshift(gcamp_y',[randi(numel(gcamp_y'),1)]);
    preds_to_test_cell = preds_cells;

    X = [ones(size(X,1),1) X];
    n = size(X,1);  % number of observations
    k = size(X,2);             % number of variables (including constant)

    b = X \ y;                 % estimate b with least squares
    u = y - X * b;             % calculates residuals
    s2 = u' * u / (n - k);     % estimate variance of error term (assuming homoskedasticity, independent observations)
    BCOV = inv(X'*X) * s2;     % get covariance matrix of b assuming homoskedasticity of error term etc...
    bse = diag(BCOV).^.5;      % standard errors

    clear Fp_vec
    for l=1:length(preds_to_test_cell)
        preds_to_test_cell{l} = preds_to_test_cell{l}+1; %because of the constant

        R = zeros(length(preds_to_test_cell{l}),k);
        for l2 = 1:length(preds_to_test_cell{l})
            R(l2, preds_to_test_cell{l}(l2))=1;
        end

        r = zeros(length(preds_to_test_cell{l}),1);          % Testing restriction: R * b = r

        num_restrictions = size(R, 1);
        F = (R*b - r)'*inv(R * BCOV * R')*(R*b - r) / num_restrictions;   % F-stat (see Hiyashi for reference)
        F_vec(l) = F;
        Fp_vec(l) = 1 - fcdf(F, num_restrictions, n - k);  % F p-val
    end

    Fp_shuff(:,i)=Fp_vec;
    F_shuff(:,i)=F_vec;

end

%% Compute p-values from shuffled distribution of F-stats
for g=1:numel(preds_cells)
    temp_sort=squeeze(sort(F_shuff(g,:)));
    temp_pval=find(F_nonshuff(g)<temp_sort); if isempty(temp_pval)==1 ;temp_pval=numel(temp_sort) ;end
    pvals(g)=1-temp_pval(1)/numel(temp_sort);
end

%% Output structure
encoding_output.coefs = b_final; %coefficients for each event 
encoding_output.pvals = pvals; %p-values for each event
encoding_output.events = cons; %behaviors tested
encoding_output.event_identiy = preds_cells; %cell of identifiers for the coefficients of each event



