classdef nnslice < nntest
  methods (Test)
    function basic(test)
      pick = @(i,x) x{i} ;
      sz = [4,5,10,10] ;
      x = test.randn(sz) ;
      slicePoints = [5] ;
      for dim = 3:4
          y = vl_nnslice(x, dim, slicePoints, []) ;
          outsizes = cellfun(@(i) size(i, dim), y) ;
          test.verifyEqual(sum(outsizes), size(x, dim)) ;
          dzdy = cellfun(@(i) test.randn(size(i)), y, 'Uni', 0) ;
          dzdx = vl_nnslice(x, dim, slicePoints, dzdy) ;

          % for gradient checking, we stack the generated derivatives and outputs
          dzdy_ = cat(dim, dzdy{:}) ;
          dzdx_ = cat(dim, dzdx{:}) ;
          test.der(@(x_) sliceWrapper(x_,dim,slicePoints, []), x, dzdy_, dzdx_, 1e-3*test.range) ;
      end
    end
  end
end

function y = sliceWrapper(x, dim, slicePoints, dzdy)
% stack output to allow numerical der comparison
tmp = vl_nnslice(x, dim, slicePoints, []) ;
y = cat(dim, tmp{:}) ;
end
