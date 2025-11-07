%% File: myEnvSwingUp.m
function envSwingUp = env_swingup()
% myEnvSwingUp - Tạo environment RL cho agent Swing-Up từ model 'rlSAC.slx
% Thông số environment
    mdl   = 'rlSAC_SwingUp';
    blk   = 'rlSAC_SwingUp/RLAgentSwingUp';

    % Observation space: alpha only
    obsInfo = rlNumericSpec([6 1], ...
        'LowerLimit', -inf(6,1), ...
        'UpperLimit', inf(6,1));
    obsInfo.Name = 'state';

    % Action space: control [-5, 5]
    actInfo = rlNumericSpec([1 1], ...
        'LowerLimit', -12, ...
        'UpperLimit', 12);
    actInfo.Name = 'action';

    %% Tạo environment Simulink
    if ~bdIsLoaded(mdl)
        load_system(mdl);
    end
    envSwingUp = rlSimulinkEnv(mdl, blk, obsInfo, actInfo);

    %% Gán hàm reset riêng biệt
    envSwingUp.ResetFcn = @resetSwingUp;
end

%% Subfunction resetSwingUp
function in = resetSwingUp(in)
    % Reset trạng thái ban đầu và setpoints cho Swing-Up agent
    alpha_init = 0;
    theta_init = 0;
    sp_theta   = 0;
    sp_alpha   = pi;

    in = setVariable(in, 'alpha_init', alpha_init);
    in = setVariable(in, 'theta_init', theta_init);
    in = setVariable(in, 'sp_theta',   sp_theta);
    in = setVariable(in, 'sp_alpha',   sp_alpha);
end