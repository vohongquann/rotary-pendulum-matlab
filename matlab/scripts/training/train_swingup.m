function statsSU = train_swingup(use_pretrain, max_episodes, max_steps_per_episode, ...
                                  score_averaging_window_length, ...
                                  stop_training_value, save_agent_value, use_parallel)
    if nargin < 6
        use_pretrain = false;
    end

    %% 0. Clear workspace and GPU
    dbclear all;
    clc;
    clearvars('-except', ...
        'use_pretrain', ...
        'max_episodes', ...
        'max_steps_per_episode', ...
        'score_averaging_window_length', ...
        'stop_training_value', ...
        'save_agent_value', ...
        'use_parallel');
    reset(gpuDevice);

    %% 1. Load the Simulink model
    mdl = 'SwingUpSimulink';
    open_system(mdl);
    Simulink.sdi.setRecordData(false);
    Simulink.sdi.clear;
    set_param(mdl, ...
        'SimulationMode','accelerator', ...
        'SimMechanicsOpenEditorOnUpdate','on');
    save_system(mdl);

    %% 2. Create RL environment and agent
    env_su = env_swingup();

    if use_pretrain
        S = load("train_models/swing_up/agent_swingup_v1.mat", "agent_su");
        agent_su = S.agent_su;
    else
        agent_su = agent_swingup(env_su);
    end

    %% 3. Assign agent to workspace and Simulink block
    assignin('base','agent_su', agent_su);
    set_param('SwingUpSimulink/RLAgentSwingUp', 'Agent', 'agent_su');

    %% 4. Prepare directories & training options
    proj = matlab.project.rootProject;

    % Main output dir
    outputDir = fullfile(proj.RootFolder, 'trained_models', 'swingup');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Checkpoint dir for automatic saves during training
    checkpointDir = fullfile(outputDir, 'saved_agents');
    if ~exist(checkpointDir, 'dir')
        mkdir(checkpointDir);
        disp(['Created checkpoint dir: ', checkpointDir]);
    end

    opts = rlTrainingOptions( ...
        MaxEpisodes                = max_episodes, ...
        MaxStepsPerEpisode        = max_steps_per_episode, ...
        ScoreAveragingWindowLength = score_averaging_window_length, ...
        StopTrainingCriteria      = "AverageReward", ...
        StopTrainingValue         = stop_training_value, ...
        Plots                     = "training-progress", ...
        UseParallel               = use_parallel, ...
        SaveAgentDirectory        = checkpointDir, ...  % 
        SaveAgentCriteria         = "EpisodeReward", ...
        SaveAgentValue            = save_agent_value ...
    );

    %% 5. Train the agent
    statsSU = train(agent_su, env_su, opts);

    %% 6. Save final agent
    savePath = fullfile(outputDir, 'agent_swingup.mat');
    save(savePath, "agent_su", "statsSU");
    disp(['Training completed. Final agent saved to: ', savePath]);
end