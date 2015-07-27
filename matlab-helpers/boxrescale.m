function boxes = boxrescale(boxes, scales)
% BOXRESCALE  Rescale box
%   BOXES = BOXRESCALE(BOXES, SCALE) rescale the BOXES by the
%   specified SCALE. BOXES is a 4 x N matrix and SCALE is either a
%   scalar or a 1 x N vector.

% Author: Andrea Vedaldi

c = [.5 0 .5 0 ; 0 .5 0 .5] * boxes ;
w = diff(boxes([1 3],:)) ;
h = diff(boxes([2 4],:)) ;
boxes = [c;c] + bsxfun(@times, [-w;-h;w;h], scales/2) ;
