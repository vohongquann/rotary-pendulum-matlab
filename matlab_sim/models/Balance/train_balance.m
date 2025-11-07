function statsBal = train_balance(use_pretrain, max_episodes, max_steps_per_episode, score_averaging_window_length,...
                                   stop_training_value, save_agent_value, use_parallel)
    % train_balance - Train a SAC agent for balancing the rotary inverted pendulum

    if nargin < 6
        use_pretrain = false;
    end

    %% 0. Clear workspace and reset GPU (except input arguments)
    dbclear all;
    clc;
    clearvars('-except', ...
        'use_pretrain', 'max_episodes', 'max_steps_per_episode', 'score_averaging_window_length',...
        'stop_training_value', 'save_agent_value', 'use_parallel');
    reset(gpuDevice);

    %% 1. Load the Simulink model in accelerator mode
    mdl = 'rlSAC_Balance';
    open_system(mdl);
    Simulink.sdi.setRecordData(false);
    Simulink.sdi.clear;
    set_param(mdl, ...
        'SimulationMode','accelerator', ...
        'SimMechanicsOpenEditorOnUpdate','on');
    save_system(mdl);

    %% 2. Create RL environment and agent
    env_b = env_balance();  % Environment with only balance dynamics

    if use_pretrain
        % Load pretrained agent if available
        S = load("models/Balance/trained_agent_b.mat", "agent_b");
        agent_b = S.agent_b;
    else
        % Create new agent from scratch
        agent_b = agent_balance(env_b);
    end

    %% 3. Assign agent to base workspace and bind to Simulink block
    assignin('base','agent_b', agent_b);
    set_param('rlSAC_Balance/RLAgentBalance', 'Agent', 'agent_b');

    %% 4. Define training options
    opts = rlTrainingOptions( ...
        MaxEpisodes                 = max_episodes, ...
        MaxStepsPerEpisode         = max_steps_per_episode, ...
        ScoreAveragingWindowLength = score_averaging_window_length, ...
        StopTrainingCriteria       = "AverageReward", ...
        StopTrainingValue          = stop_training_value, ...
        Plots                      = "training-progress", ...
        UseParallel                = use_parallel, ...
        SaveAgentDirectory         = "models/Balance/savedAgentsB", ...
        SaveAgentCriteria          = "EpisodeReward", ...
        SaveAgentValue             = save_agent_value ...
    );

    %% 5. Start training the agent
    statsBal = train(agent_b, env_b, opts);

    %% 6. Save trained agent and training statistics
    save("models/Balance/trained_agent_b.mat", "agent_b", "statsBal");
    export_weights_balance("models/Balance/trained_agent_b.mat", "models/Balance/get_action_b.m")
    disp("Finished training the Balance agent!");
end
