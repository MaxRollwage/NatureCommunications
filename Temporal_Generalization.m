clear all
close all
Subjects_initial= {'106' '107' '108' '110' '114' '115' '118' '122' '123' '126' '128' '138' '139' '142' '144' '145' '146' '148' '149' '150' '152' '156' '157' '158' '160' '162' '165'};%
AUROC_Acc=2; % decide whether 
%% define behavioral exclusion criteria
index_good_subject=[];
for sj=1:length(Subjects_initial)
    %% load directory including behavioral data from the MEG experiment
    cd(['D:\Neural Stability Pilot\GitHub\data\Study3\BehavioralData\'])
    load(['behavioral_trigger_'  Subjects_initial{sj}  '.mat'])
    %% assign behavioral variables
    Conf=behT.Confidence_initial;
    Decision=behT.Initial_Type1_decision;
    %% exclude bad trials (i.e. no response given in time)

    Conf(behT.Bad_trials)=[];
    Decision(behT.Bad_trials)=[];

    %% calculate confidence and decision bias to ensure there are enough trials of each category to train the calssifiers
    Confidence_all(sj)= nanmean(Conf);
    Left_right_bias_all(sj)= nanmean(Decision);
    if Confidence_all(sj)>1.2 & Confidence_all(sj)<1.8 & Left_right_bias_all(sj)>1.2 & Left_right_bias_all(sj)<1.8
        index_good_subject=[index_good_subject sj];
    end
    
end

%% exlude subjects that don't match criteria
for sj=1:length(index_good_subject)
    
    Subjects{sj}=Subjects_initial{index_good_subject(sj)};
end

%% load directory with MEG pre-processed temporal generalization data
cd('D:\Neural Stability Pilot\GitHub\data\Study3\TemporalGeneralization');
fPath = cd('D:\Neural Stability Pilot\GitHub\data\Study3\TemporalGeneralization');

%% Load temporal generalization data for each participant
for sj=1:length(Subjects)
    
    
    List_LR = dir(fullfile(fPath, ['Classification_LR_Generalization_change_and_confidence' Subjects{sj} '*']));
    pause(1)
    file_LR=List_LR.name;
    load(file_LR, 'AUC_gen_no_low_all','AUC_gen_no_high_all', 'AUC_gen_change_low_all', 'AUC_gen_change_high_all', 'Acc_gen_no_low_all', 'Acc_gen_change_low_all','Acc_gen_no_high_all',  'Acc_gen_change_high_all');
    
    
    if AUROC_Acc==1
        Real_no_low=Acc_gen_no_low_all;
        Real_change_low=Acc_gen_change_low_all;
        Real_no_high=Acc_gen_no_high_all;
        Real_change_high=Acc_gen_change_high_all;
     else
        Real_no_low=AUC_gen_no_low_all;
        Real_change_low=AUC_gen_change_low_all;
        Real_no_high=AUC_gen_no_high_all;
        Real_change_high=AUC_gen_change_high_all;
    end
    clear AUC_gen_no_low_all AUC_gen_change_low_all AUC_gen_no_high_all AUC_gen_change_high_all Acc_gen_no_low_all Acc_gen_change_low_all Acc_gen_no_high_all Acc_gen_change_high_all
    


    % calculate the contrast of high versus low confidence for each participant 
    Main_effect_confidence=[Real_no_high(20:end,  137:222)-Real_no_low(20:end,  137:222)]+[Real_change_high(20:end,  137:222)-Real_change_low(20:end,  137:222)];
    Main_effect_confidence_sub(sj, :, :)=Main_effect_confidence;

    clear Main_effect_confidence Real_change_high Real_no_high Real_change_low Real_no_low
    
end


Mean_Main_effect_confidence=nan(size(Main_effect_confidence_sub,2), size(Main_effect_confidence_sub,3));
Std_Main_effect_confidence=nan(size(Main_effect_confidence_sub,2), size(Main_effect_confidence_sub,3));


%% calculate the mean and standard deviation (across participants) for each contrast at every time point
for dim_train=1:size(Main_effect_confidence_sub,2)
    for  dim_test=1:size(Main_effect_confidence_sub,3)
            Mean_Main_effect_confidence(dim_train, dim_test)=nanmean(Main_effect_confidence_sub(:, dim_train, dim_test));
            Std_Main_effect_confidence(dim_train, dim_test)=sqrt(nanvar(Main_effect_confidence_sub(:, dim_train, dim_test)));
    end
end

t_crit=tinv(.975, size(Main_effect_confidence_sub, 1)-1);

addpath(genpath('D:\Documents\MATLAB\SPM\spm12'))

perm_disttribution_main_conf=[];
%% calculate permutated mean and standard deviation (across participants) for each contrast at every time point

for permutations=1:1000
  
    Perm_main_effect_confidence=nan(size(Main_effect_confidence_sub,2), size(Main_effect_confidence_sub,3));
    Perm_Std_main_effect_confidence=nan(size(Main_effect_confidence_sub,2), size(Main_effect_confidence_sub,3));

    flip_sign1=[];
    flip_sign1=randi([2], size(Main_effect_confidence_sub, 1), 1);
    flip_sign1(flip_sign1==2)=-1;
    
   % if there is no systematic difference between high and low confidence,
   % then it should not make a difference whether we look at high-low or low-high
   % by randomly flipping the sign of the high-low contrast we derive a
   % null-distribution for this contrast

    for dim_train=1:size(Main_effect_confidence_sub,2)
        for  dim_test=1:size(Main_effect_confidence_sub,3)
            Perm_main_effect_confidence(dim_train, dim_test)=nanmean((Main_effect_confidence_sub(:, dim_train, dim_test).*flip_sign1));
            Perm_Std_main_effect_confidence(dim_train, dim_test)=sqrt(nanvar((Main_effect_confidence_sub(:, dim_train, dim_test).*flip_sign1)));
        end
    end
    
    % claculate t-values for each timepoint
    perm_t_value_main_conf=(Perm_main_effect_confidence)./(Perm_Std_main_effect_confidence/sqrt(size(Main_effect_confidence_sub, 1)));
    % find time-points that individually exceed significance 
    perm_sig_cluster_main_conf=zeros(size(perm_t_value_main_conf));
    perm_sig_cluster_main_conf(abs(perm_t_value_main_conf)>t_crit)=1;

    % find clusters of significant activity
    [perm_L_main_conf,perm_NUM_main_conf] = spm_bwlabel(perm_sig_cluster_main_conf,18);
    perm_t_sum_main_conf=[];

    % sum t-values of each cluster
    for clusters=1:perm_NUM_main_conf
        ind_cluster=find(perm_L_main_conf==clusters);
        perm_t_sum_main_conf(clusters) =sum(perm_t_value_main_conf(ind_cluster));
    end

    % save summed t-values to derive a null-distribution of our test
    % statistic of interest
    perm_disttribution_main_conf=[perm_disttribution_main_conf perm_t_sum_main_conf];
    
end

% calculate t-values for each time point
t_value_main_confidence=(Mean_Main_effect_confidence)./(Std_Main_effect_confidence/sqrt(size(Main_effect_confidence_sub, 1)));
% find individual time points that exceed significance
sig_cluster_main_conf=zeros(size(t_value_main_confidence));
sig_cluster_main_conf(abs(t_value_main_confidence)>t_crit)=1;


% derive cut-off values from the null-distribution for when a cluster shows
% signifance (separate for positive and negative values as the distribution
% might be asymmetric
Cut_off_main_effect_plus = quantile(perm_disttribution_main_conf(perm_disttribution_main_conf>0),.975)
Cut_off_main_effect_minus = quantile(perm_disttribution_main_conf(perm_disttribution_main_conf<0),.025)

% find significant clsuters
[L_main_conf,NUM_main_conf] = spm_bwlabel(sig_cluster_main_conf,18)

% sum t-values of significant clusters and see if they exceed cut-off

for clusters=1:NUM_main_conf
    ind_cluster=find(L_main_conf==clusters)
    t_sum_main_conf(clusters) =sum(t_value_main_confidence(ind_cluster));
    sig_clusters_main_conf=find(t_sum_main_conf>Cut_off_main_effect_plus | t_sum_main_conf<Cut_off_main_effect_minus );
end


% define indices for significant clusters
correct_cluster_main_conf=zeros(size(t_value_main_confidence));
for clusters=1:length(sig_clusters_main_conf)
    sig_cluster=sig_clusters_main_conf(clusters);
    ind_cluster=find(L_main_conf==sig_cluster);
    correct_cluster_main_conf(ind_cluster)=1;
end




%% Plot Figure 4D

figure(1)
hold on
heatmap(t_value_main_confidence)
contour(correct_cluster_main_conf,1,  '-k', 'Linewidth', 1);
hcb=colorbar
title(hcb,'t-value'); % you can also put it above the color bar if you prefer
set(gca,'YDir','normal', 'FontSize', 15,'FontWeight','bold', 'yTick',[1 11 16 21 41 61 81], 'YTickLabel',{'0','100','150','200','400','600','800'}, 'XTick',[1 7 21 41 61 81], 'XTickLabel',{'0','50','200','400','600','800'})
fix_xticklabels(gca,2,{'FontSize',15,'FontName','Arial','FontWeight','bold'});
xlim([1 86])
ylim([1 86])
plot([1 86], [1 86], 'k-')
plot([36 36], [1 36], 'k--')
plot([1 36], [36 36], 'k--')
axis('square')
ylabel('Training time (ms)')
xlabel('Generalization time (ms)')
caxis([-3.2 3.2])


