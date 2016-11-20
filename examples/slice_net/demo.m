% run forward pass for simple networks
models = {'depth-1', 'rgb-2'} ;

for i = 1:numel(models)
    model = models{i} ;
    inputVar = 'data' ;

    %% load model
    net = cnn_init('model', model) ;
    
    % load sample image
    im = single(imread('peppers.png')) ;
    
    % resize to match network input
    imsz = net.meta.normalization.imageSize(1:2) ;
    data = imresize(im, imsz) ;
    
    % run forward pass (saving the final layer)
    net.vars(end).precious = true ;
    net.eval({inputVar, data}) ;
    
    lastVar = net.vars(end).value ;
    lastVarName = net.vars(end).name ;
    
    % print summary
    fprintf('output of network %s is %s \n', model, lastVarName) ;
    fprintf('%s has shape %s \n', lastVarName, mat2str(size(lastVar))) ;
end

% run forward pass for fusion networks
models = {'fusion-3', 'merged-4'} ; %
outputVars = {'fc2_fusion', 'fc1_fusion'} ;
for i = 1:numel(models)
    model = models{i} ;
    inputVar = 'data_RGBD' ;
    outputVar = outputVars{i} ;

    % load model
    net = cnn_init('model', model) ;

    % load sample image
    im = single(imread('peppers.png')) ;

    % resize to match network input
    imsz = net.meta.normalization.imageSize(1:2) ;
    data = imresize(im, imsz) ;

    % stack twice for fusion networks
    data = cat(3, data, data) ;

    % run forward pass (saving the final layer)
    varIndex = net.getVarIndex(outputVar) ;
    net.vars(varIndex).precious = true ;
    net.eval({inputVar, data}) ;

    lastVar = net.vars(varIndex).value ;
    lastVarName = net.vars(varIndex).name ;

    % print summary
    fprintf('output of network %s is %s \n', model, lastVarName) ;
    fprintf('%s has shape %s \n', lastVarName, mat2str(size(lastVar))) ;
end
