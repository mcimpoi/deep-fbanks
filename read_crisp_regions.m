function regions = read_crisp_regions(filePath, imageSize)

% convert RGB mask into one where each region has an integer from 1 to max
% regions
mask_ = imread(filePath) ;
[~,~,mask] = unique(reshape(mask_, [], size(mask_,3)),'rows') ;
mask = reshape(mask, [size(mask_,1) size(mask_,2)]) ;

% make sure you read the {name}_l.png file --> edges should are set to 0
% make the boundary of the rsegions equal to zero
mask(mask == max(mask(:))) = 0 ;

% make sure that the mask has the same dimension of the image
mask = imresize(mask, imageSize, 'nearest') ;

% compute areas
areas = accumarray(mask(:)+1,1,[max(mask(:))+1 1])' ;

% convert to a region list
regions.basis = mask ;
regions.labels = num2cell(1:max(mask(:))) ;
regions.scores = zeros(1, numel(regions.labels)) ;
regions.areas = areas(2:end) ;

% after resizing some regions could be entirely lost; we should remove
% these from the list
ok = regions.areas > 0 ;
if any(~ok)
  regions.areas(~ok) = [] ;
  regions.labels(~ok) = [] ;
  regions.scores(~ok) = [] ;
end



