%% Rollwage et al. (2020) Confidence drives a neural confirmation bias

%% Reproduce Figure 1B-D
clear all
close all

cd('D:\Neural Stability Pilot\GitHub\data\Study1') % change the current folder to the data directory 
Subjects = {'105' '106' '107' '108' '301' '303' '304' '305' '306' '307' '309' '310' '311' '312' '313' '314' '315' '316' '317' '318' '319' '320' '321' '322' '323' '324' '326' '329' '331'  '333' '334' '336' '337' '338' '339' '340' '342'};%


for sj= 1:length(Subjects)

    %% Load participants data
    load(['Behavioral_Data_sub_' Subjects{sj} '_2.mat'])
   
    %% Define the variables that we will use for the analysis

    % Define varibales about the stimulus
    PE_condition=locDATA2.coherence_pre;
    Motion_direction=locDATA2.dot_direction;
    Coherence_Post=locDATA2.coherence_post; 
    
    % Define variables about initial decision
    Initial_Type1_decision=locDATA2.initial_Type1_decision; 
    RT_Initial_Type1_decision=locDATA2.RT_initial_Type1_decision; 
    Accuracy_initial=locDATA2.accuracy_initial; 
    RT_Initial_Type2_decision=locDATA2.RT_initial_Type2_decision; 
    Confidence_initial=locDATA2.initial_Confidence; 
    
    %Load information about subjects final decision
    Final_Type1_decision=locDATA2.final_Type1_decision; 
    RT_final_Type1_decision=locDATA2.RT_final_Type1_decision; 
    Accuracy_final=locDATA2.accuracy_final;
    RT_final_Type2_decision=locDATA2.RT_final_Type2_decision;
    Confidence_final=locDATA2.final_Confidence; 
    
    %% Define trials trials without an answer (i.e. participants were too slow)
    index_good_trials=find(Initial_Type1_decision~=0 & Final_Type1_decision~=0);
 
    %% Delete trials without answer
    PE_condition=PE_condition(index_good_trials);
    Motion_direction=Motion_direction(index_good_trials);
    Coherence_Post=Coherence_Post(index_good_trials);
    Initial_Type1_decision=Initial_Type1_decision(index_good_trials);
    RT_Initial_Type1_decision=RT_Initial_Type1_decision(index_good_trials);
    Accuracy_initial=Accuracy_initial(index_good_trials);
    RT_Initial_Type2_decision=RT_Initial_Type2_decision(index_good_trials);
    Confidence_initial=Confidence_initial(index_good_trials);
    Final_Type1_decision=Final_Type1_decision(index_good_trials);
    RT_final_Type1_decision=RT_final_Type1_decision(index_good_trials);
    Accuracy_final=Accuracy_final(index_good_trials);
    RT_final_Type2_decision=RT_final_Type2_decision(index_good_trials);
    Confidence_final=Confidence_final(index_good_trials);
    
    %% round confidence rating value to 3 decimals
    Confidence_initial=round(Confidence_initial*1000)/1000
    Confidence_final=round(Confidence_final*1000)/1000
    
    %% Create indices for high and low positive evidence condition
    
    index_Pre_LPE=find(PE_condition==1);
    index_Pre_HPE=find(PE_condition==2);    
    
    %% find trials in which participants changed their minds
    change_of_mind=zeros(1,length(Coherence_Post))
    change_of_mind(Initial_Type1_decision==1 & Final_Type1_decision==2)=1
    change_of_mind(Initial_Type1_decision==2 & Final_Type1_decision==1)=1

    
    %% caluclate how often each confidence rating was used (to define exclusion criteria)
    Possible_confidence_ratings=round(linspace(.5, 1, 7).*1000)./1000
        for i=1:7

            used_confidence_initial(i)=length(find(Confidence_initial==Possible_confidence_ratings(i)))/length(Confidence_initial);
            
        end

    variability_confidenceratings_initial(sj)=max(used_confidence_initial);
    
    %% Do the Analysis: i.e. calculate performance, confidence etc. 
    Percentage_change_mind_HPE(sj)=length(find(change_of_mind(index_Pre_HPE)==1))/length(change_of_mind(index_Pre_HPE));
    Percentage_change_mind_LPE(sj)=length(find(change_of_mind(index_Pre_LPE)==1))/length(change_of_mind(index_Pre_LPE));
    
    
    Mean_Confidenc_LPE(sj)=mean(Confidence_initial(index_Pre_LPE));
    Mean_Confidenc_HPE(sj)=mean(Confidence_initial(index_Pre_HPE));

    Mean_RT_LPE(sj)=mean(RT_Initial_Type1_decision(index_Pre_LPE));
    Mean_RT_HPE(sj)=mean(RT_Initial_Type1_decision(index_Pre_HPE));    
    
    RT_initial_sj(sj)=mean(RT_Initial_Type1_decision);
    RT_final_sj(sj)=mean(RT_final_Type1_decision);

    Accuracy_initial_sj(sj)=mean(Accuracy_initial);
    Accuracy_final_sj(sj)=mean(Accuracy_final);

    Accuracy_initial_LPE(sj)=mean(Accuracy_initial(index_Pre_LPE));
    Accuracy_initial_HPE(sj)=mean(Accuracy_initial(index_Pre_HPE));

    Percentage_change_mind_corr(sj)=length(find(change_of_mind(Accuracy_initial==1)==1))/length(find(Accuracy_initial==1));
    Percentage_change_mind_incorr(sj)=length(find(change_of_mind(Accuracy_initial==0)==1))/length(find(Accuracy_initial==0));


    %% save trial-by-trial data for each participant for conducting multilevl mediation 
    PE{sj}=PE_condition';
    Confidence_mediation{sj}=Confidence_initial';
    Change_of_mind_mediation{sj}=change_of_mind';
    RT_mediation{sj}=log(RT_Initial_Type1_decision)';
    Accuracy_mediation{sj}=Accuracy_initial';
    Post_mediation{sj}=Coherence_Post';
    direction_mediation{sj}=Initial_Type1_decision';

    interaction_mediation1{sj}=Accuracy_initial'.*Coherence_Post';
    
    
    
end
%% define the exclusion criteria 
index_inclusion_criteria=find(Accuracy_initial_LPE>=.55 &Accuracy_initial_LPE<.875 & Accuracy_initial_HPE>=.55 &Accuracy_initial_HPE<.875 & variability_confidenceratings_initial<.9); 

%% exclude participants
Accuracy_initial_LPE=Accuracy_initial_LPE(index_inclusion_criteria);
Accuracy_initial_HPE=Accuracy_initial_HPE(index_inclusion_criteria);
Mean_Confidenc_LPE=Mean_Confidenc_LPE(index_inclusion_criteria);
Mean_Confidenc_HPE=Mean_Confidenc_HPE(index_inclusion_criteria);
Mean_RT_LPE=Mean_RT_LPE(index_inclusion_criteria);
Mean_RT_HPE=Mean_RT_HPE(index_inclusion_criteria);
RT_initial_sj=RT_initial_sj(index_inclusion_criteria);
RT_final_sj=RT_final_sj(index_inclusion_criteria);
Accuracy_initial_sj=Accuracy_initial_sj(index_inclusion_criteria);
Accuracy_final_sj=Accuracy_final_sj(index_inclusion_criteria);
Percentage_change_mind_corr=Percentage_change_mind_corr(index_inclusion_criteria);
Percentage_change_mind_incorr=Percentage_change_mind_incorr(index_inclusion_criteria);
Percentage_change_mind_HPE=Percentage_change_mind_HPE(index_inclusion_criteria);
Percentage_change_mind_LPE=Percentage_change_mind_LPE(index_inclusion_criteria);
manipulation_effect_confidence=Mean_Confidenc_HPE-Mean_Confidenc_LPE;
manipulation_effect_change_mind=Percentage_change_mind_HPE-Percentage_change_mind_LPE;

    

%% Do the statistics
[h,p, ci, stats] =ttest(Accuracy_initial_LPE, Accuracy_initial_HPE)
[h,p, ci, stats] =ttest(Mean_RT_LPE, Mean_RT_HPE)
[h,p, ci, stats] =ttest(Mean_Confidenc_LPE, Mean_Confidenc_HPE)
[r, p]=corr(manipulation_effect_confidence', manipulation_effect_change_mind')



%% Create Figure 1B
figure(1)
hold on
bar([1], [mean(Accuracy_initial_LPE)],'y', 'BarWidth', .5)
bar([2], [mean(Accuracy_initial_HPE)],'b', 'BarWidth', .5)
for ind=1:length(Accuracy_initial_LPE)
    plot([1, 2], [Accuracy_initial_LPE(ind), Accuracy_initial_HPE(ind)], 'k-')
end
plot(repmat(1, 1, length(Accuracy_initial_LPE)), Accuracy_initial_LPE,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
plot(repmat(2, 1, length(Accuracy_initial_HPE)), Accuracy_initial_HPE,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
ylim([.5 .9])
xlim([.3 2.8])
pbaspect([.6 1 1])
errorbar([1, 2],[mean(Accuracy_initial_LPE) mean(Accuracy_initial_HPE)],[std(Accuracy_initial_LPE-Accuracy_initial_HPE)/sqrt(length(Accuracy_initial_LPE)),std(Accuracy_initial_LPE-Accuracy_initial_HPE)/sqrt(length(Accuracy_initial_LPE))],'.k','LineWidth',1.5)
set(gca, 'FontSize', 17,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[1,2], 'XTickLabel',{'LPE','HPE'})
ylabel('Performance (% correct)')

%% Create Figure 1C

figure(2)
bar([1], [mean(Mean_Confidenc_LPE)],'y', 'BarWidth', .5)
hold on
bar([2], [mean(Mean_Confidenc_HPE)],'b', 'BarWidth', .5)
for ind=1:length(Accuracy_initial_LPE)
    plot([1, 2], [Mean_Confidenc_LPE(ind), Mean_Confidenc_HPE(ind)], 'k-')
end
plot(repmat(1, 1, length(Accuracy_initial_LPE)), Mean_Confidenc_LPE,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
plot(repmat(2, 1, length(Accuracy_initial_HPE)), Mean_Confidenc_HPE,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
errorbar([1, 2],[mean(Mean_Confidenc_LPE) mean(Mean_Confidenc_HPE)],[std(Mean_Confidenc_LPE-Mean_Confidenc_HPE)/sqrt(length(Mean_Confidenc_LPE)), std(Mean_Confidenc_LPE-Mean_Confidenc_HPE)/sqrt(length(Mean_Confidenc_HPE))],'.k','LineWidth',1.5)
ylim([.6 1])
xlim([.3 2.8])
pbaspect([.6 1 1])
set(gca, 'FontSize', 17,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[1,2], 'XTickLabel',{'LPE','HPE'})
ylabel('Confidence rating')


%% Create Figure 1D
figure(3)
hold on
scatter(manipulation_effect_confidence, manipulation_effect_change_mind, 'o', 'filled')
lsline
 ylabel('Manipulation effect on confidence')
 xlabel('Manipulation effect on changes of mind')
set(findall(gca, 'Type', 'Line'),'LineWidth',3)
set(gca, 'FontSize', 16,'FontName','Arial','FontWeight','bold','box','off')
% plot([-.3 1], [-.3 1],'k-')
plot([0 0], [-.3 1], 'k:')
plot([-.3 1], [0 0], 'k:')
xlim([-.08 .15])
ylim([-.16 .16])
plot(manipulation_effect_confidence(find(manipulation_effect_confidence<0)), manipulation_effect_change_mind(find(manipulation_effect_confidence<0)),'o','MarkerSize',5, 'MarkerEdgeColor',[1 .5 .3],'MarkerFaceColor',[1 .5 .3])

%%%%%%%%%%%%%%
%%%%%%%%%%%%%%
%%%%%%%%%%%%%%
%% Supplementary analysis
%%%%%%%%%%%%%%
%%%%%%%%%%%%%%

%% Create Supplementary Figure 1A
figure(4)
bar([1], [mean(Accuracy_initial_sj)],'FaceColor',[.7, .7, .7],'BarWidth', .5)
hold on
bar([2], [mean(Accuracy_final_sj)],'FaceColor',[.2, .2, .2], 'BarWidth', .5)
hold on
for ind=1:length(Accuracy_initial_LPE)
    plot([1, 2], [Accuracy_initial_sj(ind), Accuracy_final_sj(ind)], 'k-')
end
plot(repmat(1, 1, length(Accuracy_initial_LPE)), Accuracy_initial_sj,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
plot(repmat(2, 1, length(Accuracy_initial_HPE)), Accuracy_final_sj,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
errorbar([1, 2],[mean(Accuracy_initial_sj) mean(Accuracy_final_sj) ],[ std(Accuracy_initial_sj-Accuracy_final_sj)/sqrt(length(Accuracy_initial_sj)),std(Accuracy_initial_sj-Accuracy_final_sj)/sqrt(length(Accuracy_initial_sj))],'.k','LineWidth',1.5)
set(gca, 'FontSize', 16,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[1,2], 'XTickLabel',{'Initial decision','Final decision'})
ylabel('Performance (%correct)')
ylim([.5 1])
xlim([.3 2.8])
pbaspect([.6 1 1])
fix_xticklabels(gca,2,{'FontSize',14,'FontName','Arial','FontWeight','bold'});


%% Create Supplementary Figure 1B
figure(5)
bar([1], [mean(RT_initial_sj)],'FaceColor',[.7, .7, .7],'BarWidth', .5)
hold on
bar([2], [mean(RT_final_sj)],'FaceColor',[.2, .2, .2], 'BarWidth', .5)
hold on
for ind=1:length(Accuracy_initial_LPE)
    plot([1, 2], [RT_initial_sj(ind), RT_final_sj(ind)], 'k-')
end
plot(repmat(1, 1, length(Accuracy_initial_LPE)), RT_initial_sj,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
plot(repmat(2, 1, length(Accuracy_initial_HPE)), RT_final_sj,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
errorbar([1, 2],[mean(RT_initial_sj) mean(RT_final_sj) ],[ std(RT_initial_sj-RT_final_sj)/sqrt(length(RT_initial_sj)),std(RT_initial_sj-RT_final_sj)/sqrt(length(RT_initial_sj))],'.k','LineWidth',1.5)
set(gca, 'FontSize', 16,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[1,2], 'XTickLabel',{'Initial decision','Final decision'})
ylabel('Reaction time')
ylim([0.3 .9])
xlim([.3 2.8])
pbaspect([.6 1 1])
fix_xticklabels(gca,2,{'FontSize',14,'FontName','Arial','FontWeight','bold'});


%% Create Supplementary Figure 1C
figure(6)
bar([1], [mean(Percentage_change_mind_corr)],'FaceColor',[0.4660, 0.6740, 0.1880],'BarWidth', .5)
hold on
bar([2], [mean(Percentage_change_mind_incorr)],'FaceColor',[0.8500, 0.3250, 0.0980], 'BarWidth', .5)
for ind=1:length(Accuracy_initial_LPE)
    plot([1, 2], [Percentage_change_mind_corr(ind), Percentage_change_mind_incorr(ind)], 'k-')
end
plot(repmat(1, 1, length(Accuracy_initial_LPE)), Percentage_change_mind_corr,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
plot(repmat(2, 1, length(Accuracy_initial_HPE)), Percentage_change_mind_incorr,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
errorbar([1, 2],[mean(Percentage_change_mind_corr) mean(Percentage_change_mind_incorr) ],[ std(Percentage_change_mind_corr-Percentage_change_mind_incorr)/sqrt(length(Percentage_change_mind_corr)),std(Percentage_change_mind_corr-Percentage_change_mind_incorr)/sqrt(length(Percentage_change_mind_corr))],'.k','LineWidth',1.5)
set(gca, 'FontSize', 16,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[1,2], 'XTickLabel',{'Initial correct','Initial incorrect'})
ylabel('Changes of mind (%)')
ylim([0 .9])
xlim([.3 2.8])
pbaspect([.6 1 1])
fix_xticklabels(gca,2,{'FontSize',14,'FontName','Arial','FontWeight','bold'});


%% Create Supplementary Figure 5A
figure(7)
bar([1], [mean(Mean_RT_LPE)],'y', 'BarWidth', .5)
hold on
bar([2], [mean(Mean_RT_HPE)],'b', 'BarWidth', .5)
for ind=1:length(Accuracy_initial_LPE)
    plot([1, 2], [Mean_RT_LPE(ind), Mean_RT_HPE(ind)], 'k-')
end
plot(repmat(1, 1, length(Accuracy_initial_LPE)), Mean_RT_LPE,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
plot(repmat(2, 1, length(Accuracy_initial_HPE)), Mean_RT_HPE,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
errorbar([1, 2],[mean(Mean_RT_LPE) mean(Mean_RT_HPE) ],[ std(Mean_RT_LPE-Mean_RT_HPE)/sqrt(length(Mean_RT_LPE)), std(Mean_RT_LPE-Mean_RT_HPE)/sqrt(length(Mean_RT_HPE))],'.k','LineWidth',1.5)
set(gca, 'FontSize', 17,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[1,2], 'XTickLabel',{'LPE','HPE'})
ylabel('Reaction time')
ylim([.3 .9])
xlim([.3 2.8])
pbaspect([.6 1 1])


%% exclude participants from the mediation analysis

for i=1:length(index_inclusion_criteria)
    PE_good{i}=PE{index_inclusion_criteria(i)};
    Confidence_mediation_good{i}=Confidence_mediation{index_inclusion_criteria(i)};
    Change_of_mind_mediation_good{i}=Change_of_mind_mediation{index_inclusion_criteria(i)};
    
    RT_mediation_good{i}=RT_mediation{index_inclusion_criteria(i)};
    Accuracy_mediation_good{i}=Accuracy_mediation{index_inclusion_criteria(i)};
    Post_mediation_good{i}=Post_mediation{index_inclusion_criteria(i)};
    direction_mediation_good{i}=direction_mediation{index_inclusion_criteria(i)};
    interaction_mediation_good1{i}=interaction_mediation1{index_inclusion_criteria(i)};

end

%% conduct the Multi-level mediation analysis (see Supplementary Figure 5B)
addpath(genpath('D:\Documents\MATLAB\CanlabCore-master')) % add path for mediation toolbox
addpath(genpath('D:\Documents\MATLAB\MediationToolbox-master')) % add path for mediation toolbox

[paths, stats] = mediation(PE_good, Change_of_mind_mediation_good, Confidence_mediation_good,'covs',[RT_mediation_good, Accuracy_mediation_good, Post_mediation_good, interaction_mediation_good1],'logit','hierarchical','boot', 'bootsamples', 200000)

