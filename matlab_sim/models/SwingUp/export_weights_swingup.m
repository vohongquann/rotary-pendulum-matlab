function export_weights_swingup(agentFile, outputFuncFile)
    % Export the trained Swing-Up SAC actor network into a MATLAB function
    % for deployment (manual forward pass).

    % Load the saved agent
    load(agentFile, 'agent_su');
    actor = getActor(agent_su);
    net = getModel(actor);  % Extract dlnetwork from actor

    % Define the names of fully connected layers to extract weights from
    fcLayerNames = ["fc1", "fc2", "meanOut", "fcLogStd"];

    W = cell(length(fcLayerNames),1);
    b = cell(length(fcLayerNames),1);
    
    for i = 1:length(fcLayerNames)
        % Extract weights and biases from each layer
        Wrow = net.Learnables(strcmp(net.Learnables.Layer, fcLayerNames(i)) & ...
                              strcmp(net.Learnables.Parameter, 'Weights'), :);
        Brow = net.Learnables(strcmp(net.Learnables.Layer, fcLayerNames(i)) & ...
                              strcmp(net.Learnables.Parameter, 'Bias'), :);
    
        if isempty(Wrow) || isempty(Brow)
            error("Layer '%s' not found in the actor network", fcLayerNames(i));
        end

        W{i} = Wrow.Value{1};
        b{i} = Brow.Value{1};
    end

    % Write the exported policy function to a .m file
    fid = fopen(outputFuncFile, 'w');
    fprintf(fid, 'function action = get_action_su(obs)\n');
    fprintf(fid, '%%#codegen\n');
    fprintf(fid, 'obs = reshape(obs, [%d,1]);\n', size(W{1},2));

    % Declare persistent variables for weights and biases
    persistentVars = arrayfun(@(i) sprintf('W%d b%d', i, i), 1:4, 'UniformOutput', false);
    fprintf(fid, 'persistent %s\n', strjoin(persistentVars, ' '));
    fprintf(fid, 'if isempty(W1)\n');

    for i = 1:4
        fprintf(fid, '  W%d = [\n%s\n  ];\n', i, matrixToStr(W{i}));
        fprintf(fid, '  b%d = [\n%s\n  ];\n', i, matrixToStr(b{i}));
    end

    fprintf(fid, 'end\n');
    fprintf(fid, 'x = obs;\n');

    % Feedforward through hidden layers
    for i = 1:2
        fprintf(fid, 'x = W%d * x + b%d;\n', i, i);
        fprintf(fid, 'x = relu(x);\n');
    end

    % Compute mean and log standard deviation
    fprintf(fid, 'mean = W3 * x + b3;\n');
    fprintf(fid, 'logStd = W4 * x + b4;\n');
    fprintf(fid, 'std = softplus(logStd);\n');

    % Sample action using reparameterization trick
    fprintf(fid, 'epsilon = zeros(size(std));\n');
    fprintf(fid, 'preAct = mean + std .* epsilon;\n');
    fprintf(fid, 'action = tanh(preAct);\n');
    fprintf(fid, 'end\n\n');

    % ReLU activation helper
    fprintf(fid, 'function y = relu(x)\n');
    fprintf(fid, 'y = max(0, x);\n');
    fprintf(fid, 'end\n\n');

    % Softplus activation helper
    fprintf(fid, 'function y = softplus(x)\n');
    fprintf(fid, 'y = log(1 + exp(x));\n');
    fprintf(fid, 'end\n\n');

    % Tanh helper (manually defined)
    fprintf(fid, 'function y = tanh(x)\n');
    fprintf(fid, 'y = (exp(x) - exp(-x)) ./ (exp(x) + exp(-x));\n');
    fprintf(fid, 'end\n');

    fclose(fid);
    fprintf('Successfully exported to file: %s\n', outputFuncFile);
end

function s = matrixToStr(M)
    % Convert a matrix into formatted MATLAB array string
    lines = arrayfun(@(i) sprintf('    %s;', ...
        strjoin(arrayfun(@(x) sprintf('% .10f', x), M(i,:), 'UniformOutput', false))), ...
        1:size(M,1), 'UniformOutput', false);
    s = strjoin(lines, '\n');
end
