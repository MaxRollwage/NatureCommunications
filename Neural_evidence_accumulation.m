
clear all
close all

Subjects_initial = {'106' '107' '108' '110' '114' '115' '118' '122' '123' '126' '128' '138' '139' '142' '144' '145' '146' '148' '149' '150' '152' '156' '157' '158' '160' '162' '165'};%

weighted=1;
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


%% load directory with MEG pre-processed classifier predictions
cd(['D:\Neural Stability Pilot\GitHub\data\Study3\NeuralEvidenceAccumulation']);
fPath = cd(['D:\Neural Stability Pilot\GitHub\data\Study3\NeuralEvidenceAccumulation']);



%% pre defining variables
Matrix_slope=[];
Matrix_intercept=[];
Matrix_subject=[];
Matrix_choice=[];
Matrix_conf=[];
Matrix_conf2=[];
Matrix_acc=[];
Matrix_acc2=[];
Matrix_direction=[];
Matrix_post=[];
Matrix_RT=[];
Matrix_RT2=[];
Matrix_unsigned_slope=[];
Matrix_unsigned_intercept=[];
Matrix_choice2=[];


left_sj=nan(length(Subjects), 86);
right_sj=nan(length(Subjects), 86);
right_out_confirm_change_sj=nan(length(Subjects), 86);
right_out_confirm_no_sj=nan(length(Subjects), 86);
right_out_disconfirm_change_sj=nan(length(Subjects), 86);
right_out_disconfirm_no_sj=nan(length(Subjects), 86);

%% Do the analsysis for each participant
for sj=1:length(Subjects)
    
    %% Load data for each participant
    List_LR = dir(fullfile(fPath, ['Classification_LR_MainDiagonal_' Subjects{sj} '*']));
    pause(1)
    file_LR=List_LR.name;
    load(file_LR,'good_trial', 'accuracy_time', 'prob_outcome_glm','ClassifierLabels', 'change_of_mind','Confidence_init', 'behT', ...
        'left','right', 'left_out_confirm_low', 'left_out_confirm_high', 'left_out_disconfirm_low', 'left_out_disconfirm_high',...
        'right_out_confirm_low', 'right_out_confirm_high', 'right_out_disconfirm_low', 'right_out_disconfirm_high', ...
        'out_confirm_low', 'out_confirm_high', 'out_disconfirm_low', 'out_disconfirm_high', ...
        'choiceframe_confirm_low' ,'choiceframe_confirm_high', 'choiceframe_disconfirm_low', 'choiceframe_disconfirm_high');
    smooth_accuracy_time=smooth(accuracy_time);
    
    %% define time point of highest decodability
    [end_point end_index]=max(smooth_accuracy_time);
    if end_index<10 % if highest decodability is within first 100ms use the next highest time point as otherwise it is difficult to investigate evidence accumulation
        [end_point end_index2]=max(smooth_accuracy_time(10:end));
        end_index=end_index2+10
    end
    
    
    acc_subjects(sj, :)=smooth_accuracy_time;
    
    start_index=1;
    intercept=[];
    slope=[];
    %% Fit trial-by-trial regression to Classifier predicitons to derive slope and intercept
    for trial=1:length(prob_outcome_glm(1,:))
        fit_trial=fitlm([1:length(prob_outcome_glm(start_index: end_index,trial))],prob_outcome_glm(start_index: end_index,trial));
        intercept(trial)=fit_trial.Coefficients.Estimate(1);
        slope(trial)=fit_trial.Coefficients.Estimate(2);
    end
    
    %% claculate choice independent slope by flipping sign for trials in which leftward motion was presented
    unsigned_slope=zscore(slope);   % since the flipping procedure requires that the slopes are centered around zero (otherwise you can not only flip the slopes) we have to z-score the slopes
    unsigned_slope(behT.Motion_direction(good_trial)==1)=-unsigned_slope(behT.Motion_direction(good_trial)==1) % flip for leftward motion
    
    %% claculate choice independent intercept by flipping sign for trials in which leftward motion was presented
    unsigned_intercept=zscore(intercept); %see above
    unsigned_intercept(behT.Motion_direction(good_trial)==1)=-unsigned_intercept(behT.Motion_direction(good_trial)==1); %see above
    
    %% save the neural slope and intercept for the hierarchical regression
    Matrix_slope=[Matrix_slope; zscore(slope)'];
    Matrix_intercept=[Matrix_intercept;  zscore(intercept)'];
    Matrix_unsigned_slope=[Matrix_unsigned_slope; unsigned_slope'];
    Matrix_unsigned_intercept=[Matrix_unsigned_intercept; unsigned_intercept'];
    
    %% change coding of behavioral varibales for usage in the hierarchical regression
    behT.Accuracy_initial( behT.Accuracy_initial==0)=-1;
    behT.Confidence_initial(behT.Confidence_initial==1)=-1;  
    behT.Confidence_initial(behT.Confidence_initial==2)=1;  
    behT.Motion_direction(behT.Motion_direction==1)=-1;
    behT.Motion_direction(behT.Motion_direction==2)=1;
    behT.Confidence_final(behT.Confidence_final==1)=0;
    behT.Confidence_final(behT.Confidence_final==2)=1;
    
    %% save the behavioral data as predictors for the hierarchical regression
    Matrix_subject=[Matrix_subject; repmat(sj, length(intercept), 1)];
    Matrix_choice=[Matrix_choice; ClassifierLabels];
    Matrix_acc=[Matrix_acc; behT.Accuracy_initial(good_trial)];
    Matrix_conf=[Matrix_conf; behT.Confidence_initial(good_trial)];
    Matrix_direction=[Matrix_direction; behT.Motion_direction(good_trial)];
    Matrix_post=[Matrix_post; behT.Coherence_Post(good_trial)];
    Matrix_RT2=[Matrix_RT2;  behT.RT_final_Type1_decision(good_trial)];
    Matrix_conf2=[Matrix_conf2; behT.Confidence_final(good_trial)];
    Matrix_acc2=[Matrix_acc2; behT.Accuracy_final(good_trial)];
    Matrix_choice2=[Matrix_choice2;  behT.Final_Type1_decision(good_trial)];
    
    
    %% save summary statistics for each condition (this was already calculate for each participant in the pre-processing
    left_sj(sj, :)=left;
    right_sj(sj, :)=right; 
    out_confirm_low_sj(sj, :)= out_confirm_low;
    out_confirm_high_sj(sj, :)=out_confirm_high;
    out_disconfirm_low_sj(sj, :)=out_disconfirm_low;
    out_disconfirm_high_sj(sj, :)=out_disconfirm_high;

    %% number of trials per participant per condition (will be used for calculating weighted means)
    amount_confirm_low(sj)=length(find(behT.Accuracy_initial(good_trial)==1 &behT.Confidence_initial(good_trial)==-1));
    amount_confirm_high(sj)=length(find(behT.Accuracy_initial(good_trial)==1 & behT.Confidence_initial(good_trial)==1));
    amount_disconfirm_low(sj)=length(find(behT.Accuracy_initial(good_trial)==-1 & behT.Confidence_initial(good_trial)==-1));
    amount_disconfirm_high(sj)=length(find(behT.Accuracy_initial(good_trial)==-1 & behT.Confidence_initial(good_trial)==1));
    
    
    highest_decodability_time(sj)=end_index;
    highest_decodability(sj)=end_point;
    
    clear prob_outcome_glm slope intercept alternative_intercept alternative_slope end_points_window
end


amount_confirm_low=amount_confirm_low/sum(amount_confirm_low);
amount_disconfirm_low=amount_disconfirm_low/sum(amount_disconfirm_low);
amount_confirm_high=amount_confirm_high/sum(amount_confirm_high);
amount_disconfirm_high=amount_disconfirm_high/sum(amount_disconfirm_high);



%% calculate group means of representations for Figure 3C and Figure 4A&B
for k=1:length(left_sj)
    
    if weighted==0 % calculate non-weighted mean 
        average_left(k)=nanmean(left_sj(:, k));
        average_right(k)=nanmean(right_sj(:, k));

        average_confirm_low(k)=nanmean(out_confirm_low_sj(:, k));
        average_confirm_high(k)=nanmean(out_confirm_high_sj(:, k));
        average_disconfirm_low(k)=nanmean(out_disconfirm_low_sj(:, k));
        average_disconfirm_high(k)=nanmean(out_disconfirm_high_sj(:, k));

    elseif weighted==1 % calculate weighted mean 
        
        average_left(k)=nanmean(left_sj(:, k));
        average_right(k)=nanmean(right_sj(:, k));
        
        average_confirm_low(k)=wmean(out_confirm_low_sj(:, k), amount_confirm_low');
        average_confirm_high(k)=wmean(out_confirm_high_sj(:, k), amount_confirm_high');
        average_disconfirm_low(k)=wmean(out_disconfirm_low_sj(:, k), amount_disconfirm_low');
        average_disconfirm_high(k)=wmean(out_disconfirm_high_sj(:, k), amount_disconfirm_high');
        
    end
    
end

% calculate the median time of highest decodability
time_highest_decodability=median(highest_decodability_time);


%% calculate the average slope for each participant in each condition
for sj=1:length(Subjects)
    M101(sj)=mean(Matrix_unsigned_slope(Matrix_conf==-1 &  Matrix_acc==1 & Matrix_subject==sj));
    M102(sj)=mean(Matrix_unsigned_slope(Matrix_conf==-1 &  Matrix_acc==-1 & Matrix_subject==sj));
    M103(sj)=mean(Matrix_unsigned_slope(Matrix_conf==1 &  Matrix_acc==1 & Matrix_subject==sj));
    M104(sj)=mean(Matrix_unsigned_slope(Matrix_conf==1 &  Matrix_acc==-1 & Matrix_subject==sj));
end


con1=wmean(M101, amount_confirm_low);
con2=wmean(M102, amount_disconfirm_low);
con3=wmean(M103, amount_confirm_high);
con4=wmean(M104, amount_disconfirm_high);

err1=std(M101)/sqrt(length(M101));
err2=std(M102)/sqrt(length(M102));
err3=std(M103)/sqrt(length(M103));
err4=std(M104)/sqrt(length(M104));

%%%%%%%%
%%%%%%%%
%% conduct hierarchical regressions
%%%%%%%%

%% Figure 3B: regression showing that neural evidence accumulation is sensitive to motion direction
input_table_direction = table(Matrix_slope, Matrix_direction, Matrix_subject);
input_table_direction.Properties.VariableNames = {'slope','direction', 'Subject'};
fit2=fitglme(input_table_direction,'slope ~ 1 +direction','Distribution','Normal')


figure(1)
hold on
fit=fitlm([1:time_highest_decodability], average_left(1:time_highest_decodability));
ypred = predict(fit,[1:time_highest_decodability]');
left=plot([1:time_highest_decodability], average_left(1:time_highest_decodability), 'Color', [70/255 70/255 70/255],'LineWidth',1.5);
plot([1:time_highest_decodability], ypred, 'k','LineWidth',1.3);
fit=fitlm([1:time_highest_decodability], average_right(1:time_highest_decodability));
ypred = predict(fit,[1:time_highest_decodability]');
right=plot([1:time_highest_decodability], average_right(1:time_highest_decodability),'Color', [180/255 180/255 180/255],'LineWidth',1.5);
plot([1:time_highest_decodability], ypred, 'k','LineWidth',1.3);
ylim([-.20 .20])
xlim([1 time_highest_decodability+1])
pbaspect([.85 1 1])
legend([left right], {'Left', 'Right'}, 'Location', 'NorthWest','box','off')
ylabel('left  - classifier DV -   right')
xlabel('post-decision period (ms)')
set(gca, 'FontSize', 16,'FontName','Arial','FontWeight','bold','box','off', 'YTick',[-.15 -.1 -.05 0 .05 .1 .15], 'XTick',[10 20 30 40 50],'XTickLabel',{'100','200', '300','400','500'})


%% Figure 3D-F: validation of markers of post-decision evidence processing
input_table_validation = table(Matrix_unsigned_slope,Matrix_unsigned_intercept,log(Matrix_RT2),Matrix_acc2,Matrix_conf2, Matrix_subject);
input_table_validation.Properties.VariableNames = {'unsigned_slope','unsigned_intercept', 'RT2','acc2','conf2', 'Subject'};

%% Figure 3D: validating effect of neural evidence accumulation on reaction time
fit3=fitglme(input_table_validation,'RT2 ~ 1 +unsigned_intercept+unsigned_slope','Distribution','Normal')

slope_validation(1)=fit3.Coefficients.Estimate(2);
intercept_validation(1)=fit3.Coefficients.Estimate(3);
slope_validation_SEM(1)=fit3.Coefficients.SE(2);
intercept_validation_SEM(1)=fit3.Coefficients.SE(3);

figure(2)
hold on
bar([1], [slope_validation(1)],'FaceColor',[148/255, 0/255, 211/255], 'BarWidth', .6)
bar([2], [intercept_validation(1)],'FaceColor',[16/255, 147/255, 16/255], 'BarWidth', .6)
errorbar([1, 2], [slope_validation(1) intercept_validation(1)],  [slope_validation_SEM(1) intercept_validation_SEM(1)], 'k.', 'LineWidth', 2)
ylabel('fixed-effect')
title('Reaction time')
xlim([.5 2.5])
set(gca, 'FontSize', 16,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[1 2], 'XTickLabel',{'Slope','Intercept'})
fix_xticklabels(gca,2,{'FontSize',16,'FontName','Arial','FontWeight','bold'});
pbaspect([.7 1 1])


%% Figure 3E: validating effect of neural evidence accumulation on accuracy

fit3=fitglme(input_table_validation,'acc2 ~ 1 +unsigned_intercept+unsigned_slope','Distribution','Binomial')

slope_validation(2)=fit3.Coefficients.Estimate(2);
intercept_validation(2)=fit3.Coefficients.Estimate(3);
slope_validation_SEM(2)=fit3.Coefficients.SE(2);
intercept_validation_SEM(2)=fit3.Coefficients.SE(3);


figure(3)
hold on
bar([1], [slope_validation(2)],'FaceColor',[148/255, 0/255, 211/255], 'BarWidth', .6)
bar([2], [intercept_validation(2)],'FaceColor',[16/255, 147/255, 16/255], 'BarWidth', .6)
errorbar([1, 2], [slope_validation(2) intercept_validation(2)],  [slope_validation_SEM(2) intercept_validation_SEM(2)], 'k.', 'LineWidth', 2)
ylabel('fixed-effect')
 title('Accuracy')
xlim([.5 2.5])
set(gca, 'FontSize', 16,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[1 2], 'XTickLabel',{'Slope','Intercept'})
fix_xticklabels(gca,2,{'FontSize',16,'FontName','Arial','FontWeight','bold'});
pbaspect([.7 1 1])



%% Figure 3F: validating effect of neural evidence accumulation on confidence

fit3=fitglme(input_table_validation,'conf2 ~ 1 +unsigned_intercept+unsigned_slope','Distribution','Binomial')

slope_validation(3)=fit3.Coefficients.Estimate(2);
intercept_validation(3)=fit3.Coefficients.Estimate(3);
slope_validation_SEM(3)=fit3.Coefficients.SE(2);
intercept_validation_SEM(3)=fit3.Coefficients.SE(3);

figure(4)
hold on
bar([1], [slope_validation(3)],'FaceColor',[148/255, 0/255, 211/255], 'BarWidth', .6)
bar([2], [intercept_validation(3)],'FaceColor',[16/255, 147/255, 16/255], 'BarWidth', .6)
errorbar([1, 2], [slope_validation(3) intercept_validation(3)],  [slope_validation_SEM(3) intercept_validation_SEM(3)], 'k.', 'LineWidth', 2)
ylabel('fixed-effect')
title('Confidence')
xlim([.5 2.5])
set(gca, 'FontSize', 16,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[1 2], 'XTickLabel',{'Slope','Intercept'})
fix_xticklabels(gca,2,{'FontSize',16,'FontName','Arial','FontWeight','bold'});
pbaspect([.7 1 1])



%% Figure 4A-C: Hierarchical regression showing that neural evidence accumulation is biased by initial decision and confidence

input_table_Confidence = table(Matrix_unsigned_slope,Matrix_unsigned_intercept, Matrix_acc, Matrix_conf,Matrix_post,  Matrix_subject);
input_table_Confidence.Properties.VariableNames = {'unsigned_slope','unsigned_intercept','acc','confidence','post','Subject'};

% conduct hierarchical regression
fit_slope=fitglme(input_table_Confidence,'unsigned_slope ~ 1 +post+confidence*acc','Distribution','Normal')
fit_intercept=fitglme(input_table_Confidence,'unsigned_intercept ~ 1 +confidence*acc','Distribution','Normal')

%% plot Figure 4C
figure(5)
hold on
bar([2], [fit_slope.Coefficients.Estimate(5)], 'BarWidth', .3)
errorbar([2], [fit_slope.Coefficients.Estimate(5)], [fit_slope.Coefficients.SE(5)], 'k' ,'LineWidth', 2)
ylim([0 .06])
xlim([1.5 2.5])
pbaspect([.6 1 1])
set(gca, 'FontSize', 18,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[2], 'XTicklabel',{'interaction effect'})

figure(6)
hold on
bar([1:2:8], [con3 con4 con1 con2], 'BarWidth', .5)
errorbar([1:2:8], [con3 con4 con1 con2 ], [err3 err4 err1 err2 ], '.k' ,'LineWidth', 2)
plot([1:2:8], [0 0 0 0], '-k', 'LineWidth', 1)
plot(repmat(1, 1, length(M103)), M103,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
plot(repmat(3, 1, length(M104)), M104,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
plot(repmat(5, 1, length(M101)), M101,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
plot(repmat(7, 1, length(M102)), M102,'o','MarkerSize',3, 'MarkerEdgeColor',[.6 .6 .6],'MarkerFaceColor',[.6 .6 .6])
xlim([0 8])
ylim([-.65 .9])
ylabel('Slope')
set(gca, 'FontSize', 16,'FontName','Arial','FontWeight','bold','box','off', 'XTick',[1:2:8], 'XTickLabel',{'High confirm','High disconfirm','Low confirm','Low disconfirm'})
fix_xticklabels(gca,2,{'FontSize',16,'FontName','Arial','FontWeight','bold'});

%% plot Figure 4A

figure(7)
hold on
fit5=fitlm([1:time_highest_decodability], average_disconfirm_high(1:time_highest_decodability))
ypred = predict(fit5,[1:time_highest_decodability]')
change=plot([1:time_highest_decodability], average_disconfirm_high(1:time_highest_decodability),'Color', [51/255 153/255 255/255],'LineWidth',2)
plot([1:time_highest_decodability], ypred, '-k','LineWidth',1.3)
fit6=fitlm([1:time_highest_decodability], average_confirm_high(1:time_highest_decodability))
ypred = predict(fit6,[1:time_highest_decodability]')
no=plot([1:time_highest_decodability], average_confirm_high(1:time_highest_decodability),'Color', [255/255 153/255 51/255],'LineWidth',2)
plot([1:time_highest_decodability], ypred, '-k','LineWidth',1.3)
ylim([-.20 .20])
xlim([1 time_highest_decodability+1])
pbaspect([.85 1 1])
legend([no change], {'confirming', 'disconfirming'}, 'Location', 'NorthWest','box','off')
title('High confidence')
ylabel('representation motion direction')
xlabel('post-decision period (ms)')
set(gca, 'FontSize', 16,'FontName','Arial','FontWeight','bold','box','off', 'YTick',[-.15 -.1 -.05 0 .05 .1 .15], 'XTick',[10 20 30 40 50],'XTickLabel',{'100','200', '300','400','500'})

%% plot Figure 4B

figure(8)
hold on
fit5=fitlm([1:time_highest_decodability], average_disconfirm_low(1:time_highest_decodability))
ypred = predict(fit5,[1:time_highest_decodability]')
change=plot([1:time_highest_decodability], average_disconfirm_low(1:time_highest_decodability),'Color', [51/255 153/255 255/255],'LineWidth',2)
plot([1:time_highest_decodability], ypred, '-k','LineWidth',1.3)
fit6=fitlm([1:time_highest_decodability], average_confirm_low(1:time_highest_decodability))
ypred = predict(fit6,[1:time_highest_decodability]')
no=plot([1:time_highest_decodability], average_confirm_low(1:time_highest_decodability),'Color', [255/255 153/255 51/255],'LineWidth',2)
plot([1:time_highest_decodability], ypred, '-k','LineWidth',1.3)
ylim([-.20 .20])
xlim([1 time_highest_decodability+1])
pbaspect([.85 1 1])
legend([no change], {'confirming', 'disconfirming'}, 'Location', 'NorthWest','box','off')
title('Low confidence')
ylabel('representation motion direction')
xlabel('post-decision period (ms)')
set(gca, 'FontSize', 16,'FontName','Arial','FontWeight','bold','box','off', 'YTick',[-.15 -.1 -.05 0 .05 .1 .15], 'XTick',[10 20 30 40 50],'XTickLabel',{'100','200', '300','400','500'})
