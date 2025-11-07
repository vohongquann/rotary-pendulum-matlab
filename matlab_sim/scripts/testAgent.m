function experience = testAgent(agentType)
% test_agent  Simulate a pretrained Balance or Swing-Up agent in its own model
%
%   expBal = test_agent('balance');
%   expSU  = test_agent('swingup');
%
% Returns an rlSimulationExperience object.

    % 1) Validate input
    agentType = validatestring(lower(agentType), {'balance','swingup'});

    % 2) Map to model, block-local name, env-factory, MAT-file, varname
    switch agentType
      case 'balance'
        mdlName    = 'rlSAC_Balance';          % your Balance-only model file
        localBlock = 'RLAgentBalance';       % EXACT name of the Agent block
        env        = myEnvBalance();           % single-agent env
        varName    = 'saved_agent';           % agentBalance saved_agent
        matFile    = 'Agent215.mat';       % where you saved it balanceAgent.mat
      case 'swingup'
        mdlName    = 'rlSAC_SwingUp';          % your Swing-Up-only model
        localBlock = 'RLAgentSwingUp';      % EXACT name of that block
        env        = myEnvSwingUp();           
        varName    = 'agentSwingUp';            %saved_agent agentSwingUp
        matFile    = 'SwingUpAgent.mat';        
    end

    % 3) Load & configure model
    if ~bdIsLoaded(mdlName)
        load_system(mdlName);
    end
    set_param(mdlName, ...
        'SimulationMode',                'accelerator', ...
        'SimMechanicsOpenEditorOnUpdate','on');

    % 4) Load pretrained agent struct
    S = load(matFile, varName);
    agent = S.(varName);

    % 5) Assign into base workspace & update block mask
    assignin('base', varName, agent);
    blockPath = mdlName + "/" + localBlock;  % forward slash only
    set_param(blockPath, 'Agent', varName);

    % 6) Simulate
    simOpts    = rlSimulationOptions('MaxSteps', 20000);
    experience = sim(env, agent, simOpts);
end
