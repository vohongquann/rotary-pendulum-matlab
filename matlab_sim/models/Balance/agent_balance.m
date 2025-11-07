function agent = agent_balance(env)
% myAgentBalance - Cấu hình SAC agent cho nhiệm vụ giữ thăng bằng

    %% 1. Thông tin môi trường
    Ts = 0.01;

    obsInfo = getObservationInfo(env);
    actInfo = getActionInfo(env);

    numObs = prod(obsInfo.Dimension);
    numAct = prod(actInfo.Dimension);

    %% 2. Xây dựng mạng actor (Gaussian Policy với 2 output)
    actorLG = layerGraph();
    % Input + shared layer
    actorLG = addLayers(actorLG, featureInputLayer(numObs, "Name", "obsIn", "Normalization", "none"));
    actorLG = addLayers(actorLG, fullyConnectedLayer(32, "Name", "fc1"));
    actorLG = addLayers(actorLG, reluLayer("Name", "relu1"));
    actorLG = addLayers(actorLG, fullyConnectedLayer(32, "Name", "fc2"));
    actorLG = addLayers(actorLG, reluLayer("Name", "relu2"));
    actorLG = addLayers(actorLG, fullyConnectedLayer(32, "Name", "fc3"));
    actorLG = addLayers(actorLG, reluLayer("Name", "relu3"));

    % Branch: mean
    actorLG = addLayers(actorLG, fullyConnectedLayer(numAct, "Name", "meanOut"));
    % Branch: log std
    actorLG = addLayers(actorLG, fullyConnectedLayer(numAct, "Name", "fcLogStd"));
    actorLG = addLayers(actorLG, softplusLayer("Name", "logStdOut"));

    % Kết nối chung
    actorLG = connectLayers(actorLG, "obsIn", "fc1");
    actorLG = connectLayers(actorLG, "fc1", "relu1");
    actorLG = connectLayers(actorLG, "relu1", "fc2");
    actorLG = connectLayers(actorLG, "fc2", "relu2");
    actorLG = connectLayers(actorLG, "relu2", "fc3");
    actorLG = connectLayers(actorLG, "fc3", "relu3");

    % Kết nối mean branch
    actorLG = connectLayers(actorLG, "relu3", "meanOut");
    % Kết nối std branch
    actorLG = connectLayers(actorLG, "relu3", "fcLogStd");
    actorLG = connectLayers(actorLG, "fcLogStd", "logStdOut");
    
    actorLG = dlnetwork(actorLG); 
    actor = rlContinuousGaussianActor(actorLG, obsInfo, actInfo, ...
        "ObservationInputNames", "obsIn", ...
        "ActionMeanOutputNames", "meanOut", ...
        "ActionStandardDeviationOutputNames", "logStdOut", ...
        "UseDevice", "gpu");

    %% 3. Xây dựng mạng critic (Twin Q-networks)
    function critic = createCritic()
        criticLG = layerGraph();
        % Branch obs
        criticLG = addLayers(criticLG, featureInputLayer(numObs, "Name", "obsIn", "Normalization", "none"));
        criticLG = addLayers(criticLG, fullyConnectedLayer(64, "Name", "fcObs"));
        criticLG = addLayers(criticLG, reluLayer("Name", "reluObs"));
        % Branch act
        criticLG = addLayers(criticLG, featureInputLayer(numAct, "Name", "actIn", "Normalization", "none"));
        criticLG = addLayers(criticLG, fullyConnectedLayer(64, "Name", "fcAct"));
        criticLG = addLayers(criticLG, reluLayer("Name", "reluAct"));
        % Combine
        criticLG = addLayers(criticLG, concatenationLayer(1,2,"Name","concat"));
        criticLG = addLayers(criticLG, fullyConnectedLayer(128, "Name", "fc1"));
        criticLG = addLayers(criticLG, reluLayer("Name", "relu1"));
        criticLG = addLayers(criticLG, fullyConnectedLayer(64, "Name", "fc2"));
        criticLG = addLayers(criticLG, reluLayer("Name", "relu2"));
        criticLG = addLayers(criticLG, fullyConnectedLayer(64, "Name", "fc3"));
        criticLG = addLayers(criticLG, reluLayer("Name", "relu3"));
        criticLG = addLayers(criticLG, fullyConnectedLayer(1, "Name", "QOut"));

        % Connections
        criticLG = connectLayers(criticLG, "obsIn", "fcObs");
        criticLG = connectLayers(criticLG, "fcObs", "reluObs");
        criticLG = connectLayers(criticLG, "reluObs", "concat/in1");
        criticLG = connectLayers(criticLG, "actIn", "fcAct");
        criticLG = connectLayers(criticLG, "fcAct", "reluAct");
        criticLG = connectLayers(criticLG, "reluAct", "concat/in2");
        criticLG = connectLayers(criticLG, "concat", "fc1");
        criticLG = connectLayers(criticLG, "fc1", "relu1");
        criticLG = connectLayers(criticLG, "relu1", "fc2");
        criticLG = connectLayers(criticLG, "fc2", "relu2");
        criticLG = connectLayers(criticLG, "relu2", "fc3");
        criticLG = connectLayers(criticLG, "fc3", "relu3");
        criticLG = connectLayers(criticLG, "relu3", "QOut");

        criticLG = dlnetwork(criticLG); 
        critic = rlQValueFunction(criticLG, obsInfo, actInfo, ...
            "ObservationInputNames", "obsIn", ...
            "ActionInputNames", "actIn", ...
            "UseDevice", "gpu");
    end
    critic1 = createCritic();
    critic2 = createCritic();

    %% 4. Tùy chọn huấn luyện
    agentOpts = rlSACAgentOptions;
    agentOpts.SampleTime = Ts;
    agentOpts.DiscountFactor = 0.99;
    agentOpts.TargetSmoothFactor = 1e-2;
    agentOpts.ExperienceBufferLength = 1e6;
    agentOpts.MiniBatchSize = 1024;
    agentOpts.NumWarmStartSteps = 1024;
    agentOpts.ActorOptimizerOptions = rlOptimizerOptions("LearnRate",8e-3);
    agentOpts.CriticOptimizerOptions = rlOptimizerOptions("LearnRate",9e-3);

    %% 5. Tạo agent
    agent = rlSACAgent(actor,[critic1 critic2],agentOpts);
end
