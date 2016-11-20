function net = cnn_int(varargin) 
%CNN_INIT initialises a cnn as a DaG

opts.model = 'depth-1.mat' ;
opts = vl_argparse(opts, varargin) ;

modelPath = fullfile(vl_rootnn, 'data/models-import', opts.model) ;
net = dagnn.DagNN.loadobj(load(modelPath)) ;
