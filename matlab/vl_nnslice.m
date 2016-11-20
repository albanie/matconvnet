function y = vl_nnslice(X, dim, slicePoint, dzdy, varargin)
%VL_NNSLICE CNN slices a single input into multiple outputs.
%
%  Y = VL_NNSLICE(X, DIM, SLICEPOINT) slices the input X
%  along dimension DIM at the points specified by SLICEPOINT, to 
%  produce a cell array of outputs Y. The number of slices is 
%  determined by the number of values contained in SLICEPOINT.
%
%  DZDX = VL_NNSLICE(X, DIM, SLICEPOINT, DZDY) computes the derivatives
%  of the block projected onto DZDY. DZDX has the same dimensions
%  as X, while the DZDY and Y are cell arrays with matching element 
%  sizes. 

assert(~isempty(slicePoint), 'At least one slice point must be specified') ;
assert(ismember(dim, [3 4]), ...
    'Currently, only slicing along channels or batches is supported') ;

y = {} ;
if isempty(dzdy)
    sliceEnds = [slicePoint size(X, 3)] ; 
    sliceStarts = [1 sliceEnds(1:end -1) + 1] ;
    slices = arrayfun(@(x,y) x:y, sliceStarts, sliceEnds, 'Uni', 0) ;
    subs = repmat({':'}, [numel(sliceStarts) 4]) ;
    subs(:,dim) = vertcat(slices) ;
    y = arrayfun(@(x) ...
            subsref(X, struct('type', '()', 'subs', {subs(x,:)})), ...
            1:size(subs,1), 'Uni', 0) ;
else
    y = {cat(dim, dzdy{:})} ;
end
