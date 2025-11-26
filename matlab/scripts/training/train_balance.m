function statsBal = train_balance(use_pretrain, max_episodes, max_steps_per_episode, ...
                                  score_averaging_window_length, ...
                                  stop_training_value, save_agent_value, use_parallel)
    % train_balance - Train a SAC agent for balancing the rotary inverted pendulum

    if nargin < 6
        use_pretrain = false;
    end

    %% 0. Clear workspace and reset GPU (except input arguments)
    dbclear all;
    clc;
    clearvars('-except', ...
        'use_pretrain', 'max_episodes', 'max_steps_per_episode', ...
        'score_averaging_window_length', 'stop_training_value', ...
        'save_agent_value', 'use_parallel');
    reset(gpuDevice);

    %% 1. Load the Simulink model
    mdl = 'BalanceSimulink';
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
        proj = matlab.project.rootProject;
        loadPath = fullfile(proj.RootFolder, 'trained_models', 'balance', 'agent_balance.mat');
        if exist(loadPath, 'file')
            S = load(loadPath, 'agent_b');
            agent_b = S.agent_b;
            disp('Loaded pretrained agent.');
        else
            warning(['Pretrained agent not found at: ', loadPath, '. Training from scratch.']);
            agent_b = agent_balance(env_b);
        end
    else
        agent_b = agent_balance(env_b);
    end

    %% 3. Assign agent to base workspace and bind to Simulink block
    assignin('base','agent_b', agent_b);
    set_param('BalanceSimulink/RLAgentBalance', 'Agent', 'agent_b');

    %% 4. Prepare output directories & training options
    proj = matlab.project.rootProject;

    % Main output directory
    outputDir = fullfile(proj.RootFolder, 'trained_models', 'balance');
    if ~exist(outputDir, 'dir')
        mkdir(outputDir);
    end

    % Checkpoint directory (for automatic saves during training)
    checkpointDir = fullfile(outputDir, 'saved_agents');
    if ~exist(checkpointDir, 'dir')
        mkdir(checkpointDir);
        disp(['Created checkpoint directory: ', checkpointDir]);
    end

    % Training options
    opts = rlTrainingOptions( ...
        MaxEpisodes                = max_episodes, ...
        MaxStepsPerEpisode         = max_steps_per_episode, ...
        ScoreAveragingWindowLength = score_averaging_window_length, ...
        StopTrainingCriteria       = "AverageReward", ...
        StopTrainingValue          = stop_training_value, ...
        Plots                      = "training-progress", ...
        UseParallel                = use_parallel, ...
        SaveAgentDirectory         = checkpointDir, ...    
        SaveAgentCriteria          = "EpisodeReward", ...
        SaveAgentValue             = save_agent_value ...
    );

    %% 5. Start training the agent
    statsBal = train(agent_b, env_b, opts);

    %% 6. Save final agent and results
    savePath = fullfile(outputDir, 'agent_balance.mat');
    save(savePath, 'agent_b', 'statsBal');
    
    % Export weights (if function exists)
    exportDir = fullfile(proj.RootFolder, 'models', 'Balance');
    if ~exist(exportDir, 'dir')
        mkdir(exportDir);
    end
    exportWeightsPath = fullfile(exportDir, 'trained_agent_b.mat');
    exportActionPath   = fullfile(exportDir, 'get_action_b.m');
    
    if exist('export_weights_balance', 'file')
        export_weights_balance(exportWeightsPath, exportActionPath);
    else
        warning('export_weights_balance function not found. Skipping weight export.');
    end

    disp(['Balance agent training completed. Saved to: ', savePath]);
end