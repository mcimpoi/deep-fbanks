function boxes = boxclip(boxes, imageSize)
% BOXCLIP Clip boxes to a given image size
%   BOXES = BOXCLIP(BOXES, IMAGESIZE)

boxes = [max(1,boxes(1:2,:)) ; ...
         min(imageSize(1),boxes(3,:)) ;
         min(imageSize(2),boxes(4,:))] ;
