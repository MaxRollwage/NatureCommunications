%% Rollwage et al. (2020) Confidence drives a neural confirmation bias

%% Reproduce Figure 2B&C
clear all 
close all
uiopen('D:\Neural Stability Pilot\GitHub\data\Study2\HDDM_parameters.csv') % you might have to load this data in by hand before running the whole script
load('D:\Neural Stability Pilot\GitHub\data\Study2\data_compare_posteriors.mat')

addpath(genpath('D:\Documents\MATLAB\dmatoolbox\')) % add path to the DMAT toolbox

v_drift=normrnd(HDDMparameters.mean(56:78), HDDMparameters.std(56:78))*.1;

sv = 0; % no trial-by-trial variation
st = 0; % no trial-by-trial variation
sz = 0; % no trial-by-trial variation

clear sim;
N=10; % number of simulated trials per iteration
accuracy_condition=[1 1 -1 -1]; % Condition 1 & 2 represent conditions with confirmatory post-decision evidence, and condition 3 & represent disconfirmatory evidence
confidence= unique(Matrix_Initial_Confidence);

%% initialize variables
RT_dist_high_no= [];
RT_dist_high_change= [];
RT_dist_low_no= [];
RT_dist_low_change= [];
RT_sim=[];
trials_dist_low_no= [];
trials_dist_low_change= [];
trials_dist_high_no= [];
trials_dist_high_change= [];
RT_sim_correct=[];
RT_sim_incorrect=[];




for condition=1:4     % loop through the 4 conditions: confidence (high vs low) x choice-consistency (confirming vs disconfirming evidence)
    for i = 1:length(v_drift)% loop over participants
        rt1_post= [];
        acc_post=[];
        for iterations=1:1000 % simulate each condition and participants 1000 times
            
            %% draw parameter value from the posterior distribution
            
            %% Parameters that were fitted hierarchically to each individual
            nondt= normrnd(HDDMparameters.mean(3:25), HDDMparameters.std(3:25)); % non decision time (for each subject)
            z_subj= normrnd(HDDMparameters.mean(28:50), HDDMparameters.std(28:50)); % starting point bias (for each subject)
            v_drift=normrnd(HDDMparameters.mean(56:78), HDDMparameters.std(56:78))*.1; % drift-rate (for each subject)
            v_bound= normrnd(HDDMparameters.mean(85:107),HDDMparameters.std(85:107))*.1; % boundary separation (for each subject)
            % note that due to difference in the coding of the HDDM and the
            % DMAT toolbox the drift-rate and boundary separation have
            % different scale
            
            %% Parameters that were fitted as fixed effects
            %Starting point dependencies
            z_confidence=normrnd(HDDMparameters.mean(51), HDDMparameters.std(51)); % dependency of starting point on confidence
            z_confirmation=normrnd(HDDMparameters.mean(52), HDDMparameters.std(52)); % dependency of starting point on initial decision
            z_interaction=normrnd(HDDMparameters.mean(53), HDDMparameters.std(53)); % dependency of starting point on interaction (confidence x initial decision)
            %Drift-rate dependencies
            v_post_coh=normrnd(HDDMparameters.mean(79), HDDMparameters.std(79)).*.1; % dependency of drift-rate post-decision evidence coherence
            v_confidence=normrnd(HDDMparameters.mean(80),HDDMparameters.std(80))*.1; % dependency of drift-rate on confidence
            v_confirmation=normrnd(HDDMparameters.mean(81),HDDMparameters.std(81))*.1; % dependency of drift-rate on initial decision
            v_interaction=normrnd(HDDMparameters.mean(82),HDDMparameters.std(82))*.1; % dependency of drift-rate on interaction (confidence x initial decision)
            %Boundary separation dependencies
            a_conf=normrnd(HDDMparameters.mean(108),HDDMparameters.std(108))*.1; % dependency of boundary separation on confidence
            
            
            for post_decision_strength=1:2 %% Loop over post-coherence strengths
                
                if post_decision_strength==1
                    
                    v_post_coh_use=v_post_coh;
                    
                elseif post_decision_strength==2
                    v_post_coh_use=v_post_coh*2;
                end
                
                %% Sample trials with high or low confidence
                index_high=find(confidence>=median_confidence(i));
                index_low=find(confidence<median_confidence(i));
                Dist_high=Matrix_Distribution_Confidence(i, index_high)/sum(Matrix_Distribution_Confidence(i, index_high));
                Dist_low=Matrix_Distribution_Confidence(i, index_low)/sum(Matrix_Distribution_Confidence(i, index_low));
                Conf_high_ind= discretesample(Dist_high, 1);
                Conf_low_ind= discretesample(Dist_low, 1);
                Conf_high= confidence(index_high(Conf_high_ind));
                Conf_low=confidence(index_low(Conf_low_ind));
                condition_accurcy_high=accuracy_condition(condition)*Conf_high;
                condition_accurcy_low=accuracy_condition(condition)*Conf_low;
                
                %Define parameters for this iteration of simulation
                a = v_bound(i);
                t = nondt(i);
                v = v_drift(i)+v_post_coh_use;
                
                if condition==1
                    z_1=z_confidence*Conf_high;
                    z_2=z_confirmation*accuracy_condition(condition);
                    z_3=z_interaction*condition_accurcy_high;
                    
                    v_1=v_confidence*Conf_high;
                    v_2=v_confirmation*accuracy_condition(condition);
                    v_3=v_interaction*condition_accurcy_high;
                    
                    a_1=a_conf*Conf_high;
                elseif condition==2
                    z_1=z_confidence*Conf_low;
                    z_2=z_confirmation*accuracy_condition(condition);
                    z_3=z_interaction*condition_accurcy_low;
                    
                    v_1=v_confidence*Conf_low;
                    v_2=v_confirmation*accuracy_condition(condition);
                    v_3=v_interaction*condition_accurcy_low;
                    
                    a_1=a_conf*Conf_low;
                    
                elseif condition==3
                    z_1=z_confidence*Conf_high;
                    z_2=z_confirmation*accuracy_condition(condition);
                    z_3=z_interaction*condition_accurcy_high;
                    
                    v_1=v_confidence*Conf_high;
                    v_2=v_confirmation*accuracy_condition(condition);
                    v_3=v_interaction*condition_accurcy_high;
                    
                    a_1=a_conf*Conf_high;
                    
                elseif condition==4
                    z_1=z_confidence*Conf_low;
                    z_2=z_confirmation*accuracy_condition(condition);
                    z_3=z_interaction*condition_accurcy_low;
                    
                    v_1=v_confidence*Conf_low;
                    v_2=v_confirmation*accuracy_condition(condition);
                    v_3=v_interaction*condition_accurcy_low;
                    
                    a_1=a_conf*Conf_low;
                    
                end
                v_use=v+v_1+v_2+v_3;
                a_use=a+a_1;
                
                z=a_use ;
                
                z_use=z*(1/(1+exp(-(z_subj(i)+z_1+z_2+z_3))));
                
                % simulate trials and save RT and accuracy
                par = [a_use t sv z_use sz st v_use];
                [rt1 acc] = simuldiff(par, N);
                rt1_post= [rt1_post rt1];
                acc_post= [acc_post acc];
                
                
            end
        end
        
        
        
        %% Exclude trials that would not be allowed in the actual task
        index_task_allowance=find(rt1_post<.2 |rt1_post>2.5);
        acc_post(index_task_allowance)=[];
        rt1_post(index_task_allowance)=[];
        
        
        %% Calculate summary statistics for each participant and each condition
        RT_incorrect_sj(condition, i)=mean(rt1_post(acc_post==0));
        RT_correct_sj(condition,i)=mean(rt1_post(acc_post==1));
        RT_sim=[RT_sim, rt1_post];
        
        RT_sim_correct=[RT_sim_correct, rt1_post(acc_post==1)];
        RT_sim_incorrect=[RT_sim_incorrect, rt1_post(acc_post==0)];
        
        RT_sj(condition,i)=median(rt1_post);
        length_correct(condition,i)=length(find(acc_post==1));
        length_incorrect(condition,i)=length(find(acc_post==0));
        performance_sj(condition,i)=mean(acc_post);
        
        if condition==1
            RT_dist_high_no= [RT_dist_high_no rt1_post(acc_post==1)];
            RT_dist_high_change= [RT_dist_high_change rt1_post(acc_post==0)];
            trials_dist_high_no= [trials_dist_high_no,repmat(i,1, length(find(acc_post==1)))];
            trials_dist_high_change= [trials_dist_high_change,repmat(i,1, length(find(acc_post==0)))];

        elseif condition==2
            RT_dist_low_no= [RT_dist_low_no rt1_post(acc_post==1)];
            RT_dist_low_change= [RT_dist_low_change rt1_post(acc_post==0)];
            trials_dist_low_no= [trials_dist_low_no,repmat(i,1, length(find(acc_post==1)))];
            trials_dist_low_change= [trials_dist_low_change,repmat(i,1, length(find(acc_post==0)))];

        elseif condition==3
            RT_dist_high_no= [RT_dist_high_no rt1_post(acc_post==0)];
            RT_dist_high_change= [RT_dist_high_change rt1_post(acc_post==1)];
            trials_dist_high_no= [trials_dist_high_no,repmat(i,1, length(find(acc_post==0)))];
            trials_dist_high_change= [trials_dist_high_change,repmat(i,1, length(find(acc_post==1)))];
       
            
        elseif  condition==4
            RT_dist_low_no= [RT_dist_low_no rt1_post(acc_post==0)];
            RT_dist_low_change= [RT_dist_low_change rt1_post(acc_post==1)];
            trials_dist_low_no= [trials_dist_low_no,repmat(i,1, length(find(acc_post==0)))];
            trials_dist_low_change= [trials_dist_low_change,repmat(i,1, length(find(acc_post==1)))];


        end
        
    end
    
    

end

w_1= length_incorrect(3,:)./(length_correct(1,:)+length_incorrect(3,:))
w_2= length_correct(1,:)./(length_correct(1,:)+length_incorrect(3,:))
sim_RT_conf_high=w_1.*RT_incorrect_sj(3, :)+w_2.* RT_correct_sj(1, :)
trial_sim_RT_conf_high=(length_correct(1,:)+length_incorrect(3,:))

w_3= length_incorrect(4,:)./(length_correct(2,:)+length_incorrect(4,:))
w_4= length_correct(2,:)./(length_correct(2,:)+length_incorrect(4,:))
sim_RT_conf_low=w_3.*RT_incorrect_sj(4, :)+w_4.* RT_correct_sj(2, :)
trial_sim_RT_conf_low=(length_correct(2,:)+length_incorrect(4,:))


w_5= length_incorrect(2,:)./(length_correct(4,:)+length_incorrect(2,:))
w_6= length_correct(4,:)./(length_correct(4,:)+length_incorrect(2,:))
sim_RT_disconf_low=w_5.*RT_incorrect_sj(2, :)+w_6.* RT_correct_sj(4, :)
trial_sim_RT_disconf_low=(length_correct(4,:)+length_incorrect(2,:))


w_7= length_incorrect(1,:)./(length_correct(3,:)+length_incorrect(1,:))
w_8= length_correct(3,:)./(length_correct(3,:)+length_incorrect(1,:))
sim_RT_disconf_high=w_7.*RT_incorrect_sj(1, :)+w_8.* RT_correct_sj(3, :)
trial_sim_RT_disconf_high=(length_correct(3,:)+length_incorrect(1,:))




%% calculate 95% confidence intervals for accuracy (both for model simulation and empirical data)
CI_acc_sim_conf=tinv(.975, length(performance_sj(1, :)))*[nanstd(performance_sj(1, :))/sqrt(length(performance_sj(1, :))) nanstd(performance_sj(2, :))/sqrt(length(performance_sj(2, :)))]
CI_acc_sim_disconf=tinv(.975, length(performance_sj(3, :)))*[nanstd(performance_sj(3, :))/sqrt(length(performance_sj(3, :))) nanstd(performance_sj(4, :))/sqrt(length(performance_sj(4, :)))]

CI_acc_data_conf=tinv(.975, length(acc_corr_High))*[nanstd(acc_corr_High)/sqrt(length(acc_corr_High)) nanstd(acc_corr_Low)/sqrt(length(acc_corr_Low))]
CI_acc_data_disconf=tinv(.975, length(acc_incorr_High))*[nanstd(acc_incorr_High)/sqrt(length(acc_incorr_High)) nanstd(acc_incorr_Low)/sqrt(length(acc_incorr_Low))]

%% Creat Figure 2B
figure(1)
hold on
sim_conf=plot([1 2], [mean(performance_sj(1:2, :), 2)], '--', 'Color',  [255/255 153/255 51/255]);
shadedErrorBar([1, 2],[mean(performance_sj(1:2, :), 2)],CI_acc_sim_conf, {'--','Color',   [255/255 153/255 51/255]})
sim_disconf=plot([1 2], [mean(performance_sj(3:4, :), 2)],'--', 'Color',   [102/255 178/255 255/255]);
shadedErrorBar([1, 2],[mean(performance_sj(3:4, :), 2)],CI_acc_sim_disconf,{'--','Color', [102/255 178/255 255/255]})
conf=plot([1 2], [mean(acc_corr_High) mean(acc_corr_Low)],'Color',   [255/255 153/255 51/255]);
errorbar([1, 2],[mean(acc_corr_High) mean(acc_corr_Low)],CI_acc_data_conf,'o','Color', [.3 .3 .3], 'MarkerSize',8,'MarkerFaceColor', [255/255 153/255 51/255],'LineWidth',1.5)
disconf=plot([1 2], [mean(acc_incorr_High) mean(acc_incorr_Low)],'Color', [102/255 178/255 255/255]);
errorbar([1, 2],[mean(acc_incorr_High) mean(acc_incorr_Low)],CI_acc_data_disconf,'o','Color', [.3 .3 .3], 'MarkerSize',8,'MarkerFaceColor',[102/255 178/255 255/255],'LineWidth',1.5)
legend([conf,disconf,sim_conf,sim_disconf], {'Initial correct','Initial incorrect', 'Model: initial correct','Model: initial incorrect'},'Location', 'southeast','box','off')
ylim([.3 1])
xlim([.9 2.1])
pbaspect([.65 1 1])
ylabel('Accuracy (%)')
set(gca, 'FontSize', 12,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[1, 2], 'XTickLabel',{'High confidence','Low confidence'})
set(findall(gca, 'Type', 'Line'),'LineWidth',1.5)
fix_xticklabels(gca,2,{'FontSize',12,'FontName','Arial','FontWeight','bold'});


%% calculate 95% confidence intervals for reaction times (both for model simulation and empirical data)
CI_RT_sim_conf=tinv(.975, length(sim_RT_conf_high))*[nanstd(sim_RT_conf_high)/sqrt(length(sim_RT_conf_high)) nanstd(sim_RT_conf_low)/sqrt(length(sim_RT_conf_low))]
CI_RT_sim_disconf=tinv(.975, length(sim_RT_disconf_high))*[nanstd(sim_RT_disconf_high)/sqrt(length(sim_RT_disconf_high)) nanstd(sim_RT_disconf_low)/sqrt(length(sim_RT_disconf_low))]
CI_RT_data_conf=tinv(.975, length(RT_conf_high))*[nanstd(RT_conf_high)/sqrt(length(RT_conf_high)) nanstd(RT_conf_low)/sqrt(length(RT_conf_low))]
CI_RT_data_disconf=tinv(.975, length(RT_disconf_high))*[nanstd(RT_disconf_high)/sqrt(length(RT_disconf_high)) nanstd(RT_disconf_low)/sqrt(length(RT_disconf_low))]

%% Creat Figure 2C
figure(3)
hold on
shadedErrorBar([1, 2],[mean(sim_RT_conf_high) mean(sim_RT_conf_low)],CI_RT_sim_conf,{'--','Color', [0 0 0]}, 1)
sim_conf=plot([1 2], [mean(sim_RT_conf_high) mean(sim_RT_conf_low)], '--', 'Color',  [0 0 0]);
shadedErrorBar([1, 2],[mean(sim_RT_disconf_high) mean(sim_RT_disconf_low)],CI_RT_sim_disconf,{'--','Color', [.6 .6 .6]}, 1)
sim_disconf=plot([1 2], [mean(sim_RT_disconf_high) mean(sim_RT_disconf_low)], '--', 'Color',   [.6 .6 .6]);
conf=plot([1 2], [mean(RT_conf_high) mean(RT_conf_low)], 'Color',  [0 0 0]);
errorbar([1, 2],[mean(RT_conf_high) mean(RT_conf_low)],CI_RT_data_conf,'o','Color', [0 0 0], 'MarkerSize',8,'MarkerFaceColor', [0 0 0],'LineWidth',1.5)
disconf=plot([1 2], [mean(RT_disconf_high) mean(RT_disconf_low)],'Color',  [.6 .6 .6]);
errorbar([1, 2],[mean(RT_disconf_high) mean(RT_disconf_low)],CI_RT_data_disconf,'o','Color', [.6 .6 .6], 'MarkerSize',8,'MarkerFaceColor',   [.6 .6 .6],'LineWidth',1.5)
legend([conf,disconf, sim_conf,sim_disconf ], {'No change','Change of mind', 'Model: no change','Model: change of mind'},'Location', 'southeast','box','off')
ylabel('Reaction time (sec)')
set(gca, 'FontSize', 12,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[1, 2], 'XTickLabel',{'High confidence','Low confidence'})
set(findall(gca, 'Type', 'Line'),'LineWidth',1.5)
ylim([.5 1.05])
xlim([.9 2.1])
pbaspect([.65 1 1])
fix_xticklabels(gca,2,{'FontSize',12,'FontName','Arial','FontWeight','bold'});


%% create the full distributions of RT for each condition
index_low_no_change_use=[];
index_high_no_change_use=[];
index_low_change_use=[];
index_high_change_use=[];
subjects=unique(Matrix_group)
index_correct_use=[]
index_incorrect_use=[]
median_confidence_use=median_confidence

for sj=1:length(v_drift)
    
    index_low_no_change=find(Matrix_Initial_Confidence<=median_confidence_use(sj)  & Matrix_Accuracy_initial==Matrix_Accuracy_final & Matrix_group==subjects(sj));
    index_high_no_change=find(Matrix_Initial_Confidence>median_confidence_use(sj) & Matrix_Accuracy_initial==Matrix_Accuracy_final & Matrix_group==subjects(sj));

    index_low_change=find(Matrix_Initial_Confidence<=median_confidence_use(sj) & Matrix_Accuracy_initial~=Matrix_Accuracy_final & Matrix_group==subjects(sj));
    index_high_change=find(Matrix_Initial_Confidence>median_confidence_use(sj)  & Matrix_Accuracy_initial~=Matrix_Accuracy_final & Matrix_group==subjects(sj));
    

    index_low_no_change_use=[index_low_no_change_use; index_low_no_change];
    index_high_no_change_use=[index_high_no_change_use; index_high_no_change];
    index_low_change_use=[index_low_change_use; index_low_change];
    index_high_change_use=[index_high_change_use; index_high_change];
    
    
end

%% separate the empirical RT into the 4 conditions
RT_low_no=Matrix_RT(index_low_no_change_use);
RT_high_no=Matrix_RT(index_high_no_change_use);
RT_low_change=Matrix_RT(index_low_change_use);
RT_high_change=Matrix_RT(index_high_change_use);


%% make sure that only data is included that aligns with the task structure (which should be the case anyway unles something went wrong with the timing of response collection)
RT_low_no(RT_low_no<.2 | RT_low_no>2.5)=[];
RT_high_no(RT_high_no<.2 | RT_high_no>2.5 )=[];
RT_low_change(RT_low_change<.2 | RT_low_change>2.5 )=[];
RT_high_change(RT_high_change<.2 | RT_high_change>2.5)=[];

%% sample data from the simulation for each participant and each condition,
%% this ensure that participants who contributed more with their empirical data in one condition contribute to the same degree with their data to the simulation
    sampled_1= [];
        sampled_2= [];
        sampled_3= [];
        sampled_4= [];

for iteration =1:20
    for i=1:23
        
        sampled_1= [sampled_1 datasample(RT_dist_low_no(trials_dist_low_no==i),trial_conf_low(i))];
        sampled_2= [sampled_2 datasample(RT_dist_high_no(trials_dist_high_no==i),trial_conf_high(i))];
        sampled_3= [sampled_3 datasample(RT_dist_high_change(trials_dist_high_change==i),trial_disconf_high(i))];
        sampled_4= [sampled_4 datasample(RT_dist_low_change(trials_dist_low_change==i),trial_disconf_low(i))];
    end
end



figure(3)
subplot(2,2,1)
[h, u] = distribution_plot(sampled_1, 'color', [1 0 0], 'alpha', .5)
hold on
[h, u] = distribution_plot(RT_low_no, [0 1 1], 'alpha', .5)
ylim([0 1.1])
title('Low confidence & no change')

subplot(2,2,3)
[h, u] = distribution_plot(sampled_2, 'color', [1 0 0], 'alpha', .5)
hold on
[h, u] = distribution_plot(RT_high_no, [0 1 1], 'alpha', .5)
ylim([0 1.1])
title('High confidence & no change')


subplot(2,2,2)
[h, u] = distribution_plot(sampled_4, 'color', [1 0 0], 'alpha', .5)
hold on
[h, u] = distribution_plot(RT_low_change, [0 1 1], 'alpha', .5)
ylim([0 1.1])
title('Low confidence & change of mind')

subplot(2,2,4)
[h, u] = distribution_plot(sampled_3, 'color', [1 0 0], 'alpha', .5)
hold on
[h, u] = distribution_plot(RT_high_change, [0 1 1], 'alpha',.5)
ylim([0 1.1])
title('High confidence & change of mind')
