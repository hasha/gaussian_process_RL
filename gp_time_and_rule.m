
function [average_rewards,learning_curve] = gp_time_and_rule(condition,sess_type,w_magnitude)

%gen_stim
load('stim3.mat')
up2 = stim1; up3 = stim2;

clearvars -except up2 up3 condition sess_type w_magnitude

if condition == 1
    w = w_magnitude;
    w_rule = w_magnitude./2;
elseif condition == 2
    w_rule = w_magnitude;
    w = w_magnitude./2;
end

%0.88

x = [10:5:60];
x_rule = [1,2];

delta = 0.1;
sigma_obs = 0.1;
num_trials = 100; %only look at 100 of the 400 trials
pe = ones(1,num_trials);
Q = ones(1,num_trials);
rule = 1; %starting rule

for iter = 1:100
    
    rand('twister', iter);
    up2 = up2(randperm(num_trials),:);
    up3 = up3(randperm(num_trials),:);

    
    stim = up2(1,:);
    init_choice = randperm(numel(x),1);
    sample_x = x(init_choice);
    sample_x_rule = 1;
    [sample_y,pc] = decision_module_time_and_rule(stim,sample_x,sample_x_rule);
    sample_y_rule = sample_y;
    
    pe = ones(1,num_trials);
    Q = ones(1,num_trials);

    beta_t = 40;
    beta_t_rule = 40;
        

    for t = 1:num_trials-1
        
        if sess_type == 1 || sess_type == 3% time switch
            if t<round(num_trials/2) %2
                stim = up2(t+1,:);
            elseif t > round(num_trials/2) %3
                stim = up3(t+1,:);
            end 
        elseif sess_type == 2 || sess_type == 4% rule switch
            stim = up2(t+1,:);
            if t<round(num_trials/2)
                rule = 1;
            elseif t > round(num_trials/2) 
                rule = 2;
                %beta_t_rule = 500;
            end
            
        end
       
        
        [pick_x] = gp(x,sample_x,sample_y,beta_t,sigma_obs);
        [pick_x_rule,bound] = gp(x_rule,sample_x_rule,sample_y_rule,beta_t_rule,sigma_obs);

        
        sample_x = [sample_x,x(pick_x)];
        sample_x_rule = [sample_x_rule,x_rule(pick_x_rule)];
        
        %beta_t = 6 * log(2 * (t * pi).^2 / (6 * delta));
        beta_t2(t) = beta_t;
        beta_t_rule2(t) = beta_t_rule;
        all_std_rule(t) = bound(2);
        
        %run decision module here
        
        rule_truth = x_rule(pick_x_rule)==rule;
        [reward,c] = decision_module_time_and_rule(stim,x(pick_x),rule_truth);
        sample_y = [sample_y,reward];
        sample_y_rule = sample_y;
        
        if sess_type == 3 %time only switch
            rule_truth = NaN;
            [reward,c] = decision_module_time_and_rule(stim,x(pick_x),rule_truth);
        elseif sess_type == 4 %rule only switch
            stim = NaN;
            [reward,c] = decision_module_time_and_rule(stim,x(pick_x),rule_truth);
        end
        
        pc(t+1) = c;

%        alpha2 = 0.8;
         pe(t+1) = reward - Q(t);
         Q(t+1) = Q(t) + 0.3*pe(t+1); 
         
         if pe(t)<0
             beta_t = beta_t + w*abs(pe(t+1));
             beta_t_rule = beta_t_rule + w_rule*abs(pe(t+1));
         end
%              
         all_rule(t) = rule_truth;
        
        
    end
    
    all_sample_x(iter,:) = sample_x;

    learning_curve(iter,:) = gaussFilterSpikes(pc,1);
    all_pe(iter,:) = pe;
    all_beta_t2(iter,:) = beta_t2;
    average_rewards(iter) = nanmean(sample_y);

    
end


end
% 

function [pick_x,bound] = gp(x,sample_x,sample_y,beta_t,sigma_obs)

    K = k_gaussian(sample_x,x)*inv(k_gaussian(sample_x, sample_x) + eye(numel(sample_x)) * sigma_obs.^2);
    mu = K*sample_y';
    sigma = k_gaussian(x,x) - K*k_gaussian(x, sample_x);
    std_1d = sqrt(diag(sigma));
    [~,pick_x] = max(mu + sqrt(beta_t) * std_1d);
    bound = sqrt(beta_t) * std_1d;
    
end
    

function k_matrix = k_gaussian(x1,x2)

% USES A RADIAL BASIS FUNCTION KERNEL

    l = 0.1;
    sigma = 1;
    x1_matrix = repmat(x1,numel(x2),1);
    x2_matrix = x2';
    k_matrix = exp(-(x1_matrix - x2_matrix).^2./(2*l*l))*sigma.^2;
    
end
% 