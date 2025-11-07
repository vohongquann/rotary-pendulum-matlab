function statsSU = train_swingup(use_pretrain, max_episodes, max_steps_per_episode, score_averaging_window_length, ...
                                  stop_training_value, save_agent_value, use_parallel)
    % train_swingup - Train the swing-up RL agent for rotary inverted pendulum

    if nargin < 6
        use_pretrain = false;
    end

    %% 0. Clear workspace and GPU
    dbclear all;  % Remove all breakpoints
    clc;          % Clear console
    clearvars('-except', ...
        'use_pretrain', 'max_episodes', 'max_steps_per_episode', 'score_averaging_window_length', ...
        'stop_training_value', 'save_agent_value', 'use_parallel');  % Keep input arguments only
    reset(gpuDevice);  % Reset GPU state

    %% 1. Load the Simulink model in accelerator mode
    mdl = 'rlSAC_SwingUp';
    open_system(mdl);  % Open the model GUI
    Simulink.sdi.setRecordData(false);  % Disable simulation logging
    Simulink.sdi.clear;                 % Clear previous simulation data
    set_param(mdl, ...
        'SimulationMode','accelerator', ...
        'SimMechanicsOpenEditorOnUpdate','on');
    save_system(mdl);  % Save model settings

    %% 2. Create RL environment and agent
    env_su = env_swingup();  % Load swing-up environment only

    if use_pretrain
        % Load pretrained agent if specified
        S = load("models/SwingUp/trained_agent_su.mat", "agent_su");
        agent_su = S.agent_su;
    else
        % Otherwise, create a new agent from scratch
        agent_su = agent_swingup(env_su);
    end

    %% 3. Assign agent to workspace and Simulink block
    assignin('base','agent_su', agent_su);  % Expose agent to base workspace
    set_param('rlSAC_SwingUp/RLAgentSwingUp', 'Agent', 'agent_su');  % Link to agent block

    %% 4. Define training options
    opts = rlTrainingOptions( ...
        MaxEpisodes                = max_episodes, ...
        MaxStepsPerEpisode        = max_steps_per_episode, ...
        ScoreAveragingWindowLength = score_averaging_window_length, ...
        StopTrainingCriteria      = "AverageReward", ...
        StopTrainingValue         = stop_training_value, ...
        Plots                     = "training-progress", ...
        UseParallel               = use_parallel, ...
        SaveAgentDirectory        = "models/SwingUp/savedAgentsSU", ...
        SaveAgentCriteria         = "EpisodeReward", ...
        SaveAgentValue            = save_agent_value ...
    );

    %% 5. Train the agent
    statsSU = train(agent_su, env_su, opts);

    %% 6. Save the trained agent and results
    save("models/SwingUp/trained_agent_su.mat", "agent_su", "statsSU");
    export_weights_swingup("models/SwingUp/trained_agent_su.mat", "models/SwingUp/get_action_su.m")
    disp("Swing-Up agent training completed!");
end
