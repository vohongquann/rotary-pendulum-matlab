function export_weights_balance(agentFile, outputFuncFile)
    % Load cả agent_b và statsBal
    S = load(agentFile, 'agent_b', 'statsBal');
    if ~isfield(S,'agent_b') || ~isfield(S,'statsBal')
        error("File %s phải chứa biến agent_b và statsBal", agentFile);
    end
    agent_b = S.agent_b;

    % Lấy actor từ agent
    actor = getActor(agent_b);
    net   = getModel(actor);  % dlnetwork

    % Các layer FC có trọng số
    fcLayerNames = ["fc1","fc2","fc3","meanOut","fcLogStd"];

    W = cell(numel(fcLayerNames),1);
    b = cell(numel(fcLayerNames),1);
    for i = 1:numel(fcLayerNames)
        L = fcLayerNames(i);
        Wrow = net.Learnables( ...
            strcmp(net.Learnables.Layer, L) & ...
            strcmp(net.Learnables.Parameter, 'Weights'), :);
        Brow = net.Learnables( ...
            strcmp(net.Learnables.Layer, L) & ...
            strcmp(net.Learnables.Parameter, 'Bias'), :);
        if isempty(Wrow) || isempty(Brow)
            error("Không tìm thấy weight/bias cho layer '%s'", L);
        end
        W{i} = Wrow.Value{1};
        b{i} = Brow.Value{1};
    end

    % Viết file .m
    fid = fopen(outputFuncFile,'w');
    fprintf(fid,'function action = PolicyBalance(obs)\n');
    fprintf(fid,'%%#codegen\n');
    fprintf(fid,'obs = reshape(obs, [%d,1]);\n\n', size(W{1},2));

    % Persistent
    pv = arrayfun(@(i) sprintf('W%d b%d',i,i), 1:5, 'UniformOutput',false);
    fprintf(fid,'persistent %s\n', strjoin(pv,' '));
    fprintf(fid,'if isempty(W1)\n');
    for i=1:5
        fprintf(fid,'  W%d = [\n%s\n  ];\n', i, matrixToStr(W{i}));
        fprintf(fid,'  b%d = [\n%s\n  ];\n', i, matrixToStr(b{i}));
    end
    fprintf(fid,'end\n\n');

    fprintf(fid,'x = obs;\n');
    for i=1:3
        fprintf(fid,'x = W%d*x + b%d;\n',i,i);
        fprintf(fid,'x = relu(x);\n');
    end
    fprintf(fid,'mean = W4*x + b4;\n');
    fprintf(fid,'logStd = W5*x + b5;\n');
    fprintf(fid,'std = softplus(logStd);\n\n');

    fprintf(fid,'% reparameterization\n');
    fprintf(fid,'epsilon = zeros(size(std));\n');
    fprintf(fid,'preAct = mean + std .* epsilon;\n');
    fprintf(fid,'action = tanh(preAct);\n');
    fprintf(fid,'end\n\n');

    % Helpers
    fprintf(fid,'function y = relu(x), y = max(0,x); end\n\n');
    fprintf(fid,'function y = softplus(x), y = log(1+exp(x)); end\n\n');
    fprintf(fid,'function y = tanh(x), y = (exp(x)-exp(-x))./(exp(x)+exp(-x)); end\n');
    fclose(fid);

    fprintf('Successfully export policy Balance → %s\n', outputFuncFile);
end

function s = matrixToStr(M)
    lines = arrayfun(@(r) sprintf('    %s;', ...
        strjoin(arrayfun(@(v) sprintf('% .10f',v), M(r,:), 'UniformOutput',false))), ...
        1:size(M,1), 'UniformOutput',false);
    s = strjoin(lines, sprintf('\n'));
end
