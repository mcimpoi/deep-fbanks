function [c, acc] = compute_confusion(numClasses, gts, preds, areas, doNotNormalizePerClass)
if ~exist('doNotNormalizePerClass')
  doNotNormalizePerClass = false ;
end
if nargin <= 3, areas = ones(size(gts)) ; end
c = accumarray([gts(:), preds(:)], areas(:), numClasses*[1,1]) ;
if ~doNotNormalizePerClass
  c = bsxfun(@times, 1./sum(c,2), c) ;
  acc = mean(diag(c)) ;
else
  c = c / sum(c(:)) ;
  acc = sum(diag(c)) ;
end
