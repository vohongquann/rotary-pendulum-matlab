function startup()
    proj = matlab.project.rootProject;
    
    % Thêm các thư mục con cần thiết vào path
    foldersToAdd = {
        fullfile(proj.RootFolder, 'cad')
        fullfile(proj.RootFolder, 'scripts')
        fullfile(proj.RootFolder, 'models')
        fullfile(proj.RootFolder, 'common')
    };

    for i = 1:numel(foldersToAdd)
        addpath(genpath(foldersToAdd{i}));
    end
    
    disp('[InvertedPendulumProject] Project initialized!');
end
