function [im,regions,gt] = os_get_gt_regions(imdb, imageId)
% Get the image and GT regions for imageId


% read the image
ii = find(imdb.images.id == imageId) ;
im = imread(fullfile(imdb.imageDir, imdb.images.name{ii})) ;

if size(im,3) == 1, im = repmat(im, [1 1 3]) ; end
height = size(im,1) ;
width = size(im,2) ;


% read all corresponding segments
sel = find(imdb.segments.imageId == imdb.images.id(ii)) ;
area = zeros(1, numel(sel)) ;
support = cell(1, numel(sel)) ;
for j = 1:numel(sel)
  if isfield(imdb.segments, 'mask') && ...
      exist(fullfile(imdb.maskDir, imdb.segments.mask{sel(j)}), 'file')
    mask = logical(imread(fullfile(imdb.maskDir, imdb.segments.mask{sel(j)}))) ;
  else
    mask = true(height, width);
  end

  if size(mask,3) > 1, mask = mask(:,:,1) ; end
  area(j) = sum(mask(:)) ;
  support{j} = mask ;
end

% subtract smaller regions from larger ones, in case they overlap
occupied = false(height, width) ;
[~,perm] = sort(area) ;
for i = 1:numel(perm)
  j = perm(i) ;
  support{j} = support{j} & ~occupied ;
  occupied = occupied | support{j} ;
end

% create region structure
regions.basis = zeros(height, width) ;
regions.labels = cell(1, numel(support)) ;
regions.area = zeros(1, numel(support)) ;
regions.segmentIndex = zeros(1, numel(support)) ;
for j = 1:numel(support)
  regions.basis(support{j}) = j ;
  regions.labels(j) = {j} ;
  regions.area(j) = sum(support{j}(:)) ;
  regions.segmentIndex(j) = sel(j) ;
end

% drop empty regions (very uncommon but crashes the code)
ok = false(1, numel(regions.labels)) ;
for r = 1:numel(regions.labels)
  ok(r) = any(ismember(regions.basis(:), regions.labels{r})) ;
end
if ~all(ok)
  warning('Dropping some empty gt regions') ;
end
regions.labels = regions.labels(ok) ;

% now create a segmentation mask with class labels for the regions
if (size(imdb.segments.label, 1) == 1)
    gt = zeros(height, width) ;
    for r = 1:numel(regions.labels)
      mask = ismember(regions.basis, regions.labels{r}) ;
      gt(mask) = imdb.segments.label(regions.segmentIndex(r)) ;
    end
else
    numLabels = size(imdb.segments.label, 1);

    gt = zeros(numLabels, height, width) ;
    for ll = 1 : numLabels
        tmp = zeros(height, width);
        for r = 1:numel(regions.labels)
            mask = ismember(regions.basis, regions.labels{r}) ;
            tmp(mask) = (imdb.segments.label(ll, regions.segmentIndex(r)) == 1);
            gt(ll, :, :) = tmp;
        end
    end
end

if 0
  figure(3) ; clf ;
  [~,gt_] = ismember(gt, find(imdb.meta.inUse)) ;
  subplot(1,2,1) ; imagesc(im) ; title(imdb.images.name{ii}) ;
  subplot(1,2,2) ; image(gt_ + 1);
  colormap([ [1,1,1]; distinguishable_colors(sum(imdb.meta.inUse))]) ;
end

end


