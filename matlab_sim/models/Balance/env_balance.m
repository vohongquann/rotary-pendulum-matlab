%% File: myEnvBalance.m
function envBalance = env_balance()
% myEnvBalance - Tạo environment RL cho agent Balance từ model 'rlSAC.slx'

    %% Thông số environment
    mdl   = 'rlSAC_Balance';
    blk   = 'rlSAC_Balance/RLAgentBalance';
    
    % Observation space: theta, theta_dot, alpha, alpha_dot, sp_theta, sp_alpha, action
    obsInfo = rlNumericSpec([6 1], ...
        'LowerLimit', -inf(6,1), ...
        'UpperLimit', inf(6,1));
    obsInfo.Name = 'state';

    % Action space: voltage [-1, 1]
    actInfo = rlNumericSpec([1 1], ...
        'LowerLimit', -12, ...
        'UpperLimit', 12);
    actInfo.Name = 'action';

    %% Tạo environment Simulink
    if ~bdIsLoaded(mdl)
        load_system(mdl);
    end
    envBalance = rlSimulinkEnv(mdl, blk, obsInfo, actInfo);

    %% Gán hàm reset riêng biệt
    envBalance.ResetFcn = @resetBalance;
end

%% Subfunction resetBalance
function in = resetBalance(in)
    % Reset trạng thái ban đầu và setpoints cho Balance agent
    alpha_init = 0;
    theta_init = 0;
    sp_theta   = (2*rand-1)*pi/2;
    sp_alpha   = pi;
    
    in = setVariable(in, 'alpha_init', alpha_init);
    in = setVariable(in, 'theta_init', theta_init);
    in = setVariable(in, 'sp_theta',   sp_theta);
    in = setVariable(in, 'sp_alpha',   sp_alpha);
end



