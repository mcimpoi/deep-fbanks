function [code, codeLoc] = get_dcnn_features(net, im, regions, varargin)
% GET_DCNN_FEATURES  Get convolutional features for an image region
%   This function extracts the DCNN (CNN+FV) for one or more regions in an image.
%   These can be used as SIFT replacement in e.g. a Fisher Vector.
%
%   MASK should be an array that has the same spatial dimensions of the
%   imge IM, but one or more planes. Each plane specifies one or more
%   non-overlapping image regions by associating to each pixel a
%   corresponding region ID. IDs should be contiguous intergers starting
%   from 1. Any pixel assogined the label 0 does not belong to any region
%   in that plane.
%
%   MASK is a relatively efficient manner of specifying multiple
%   segmentations of the image.
%
%   Note that both IM and MASK are cell arrays, allowing to process
%   a number of images at the same time.

opts.useSIFT = false ;
opts.crop = true ;
%opts.scales = [0.5 0.75 1] ; %CUB
opts.scales = 2.^(1.5:-.5:-3); % as in CVPR14 submission
opts.encoder = struct();
opts.numSpatialSubdivisions = 1 ;
opts.maxNumLocalDescriptorsReturned = +inf ;
opts = fix_vl_argparse(opts, varargin) ;

if (numel(opts.numSpatialSubdivisions) == 1)
    opts.numSpatialSubdivisions = opts.numSpatialSubdivisions * [1 1];
end

% Find geometric parameters of the representation. x is set to the
% leftmost pixel of the receptive field of the lefmost feature at
% the last level of the network. This is computed by backtracking
% from the last layer. Then we obtain a map
%
%   x(u) = offset + stride * u
%
% from a feature index u to pixel coordinate v of the center of the
% receptive field.

if opts.useSIFT
  binSize = 8;
  offset = 1 + 3/2 * binSize ;
  stride = 4;
  border = binSize*2 ;
else
  info = fix_vl_simplenn_display(net) ;
  %vl_simplenn_display(net) ;
  x=1 ;
  %
  for l=numel(net.layers):-1:1
    x=(x-1)*info.stride(2,l)-info.pad(2,l)+1 ;
  end
  offset = round(x + info.receptiveFieldSize(end)/2 - 0.5);
  stride = prod(info.stride(1,:));
  border = ceil(info.receptiveFieldSize(end)/2 + 1);
  averageColour = mean(mean(net.normalization.averageImage,1),2) ;
end

if ~iscell(im)
  im = {im} ;
  regions = {regions} ;
end

numNull = 0 ;
numReg = 0 ;

% for each image
for k=1:numel(im)
  % crop region
  im_w = size(im{k}, 1);
  im_h = size(im{k}, 2);
  min_ratio = min(im_w / (border + 2), im_h / (border + 2));
  if (min_ratio < 1)
      im{k} = imresize(im{k}, 1 / min_ratio);
      regions{k}.basis = imresize(regions{k}.basis, 1/min_ratio, 'nearest');
  end
  [im_cropped, regions_cropped] = crop(opts, single(im{k}), regions{k}, border) ;

  crop_h = size(im_cropped,1) ;
  crop_w = size(im_cropped,2) ;
  psi = cell(1, numel(regions_cropped.labels)) ;
  loc = cell(1, numel(regions_cropped.labels)) ;
  res = [] ;

  % for each scale
  for s=1:numel(opts.scales)
    % 30 was added to prevent crash on 2nd conv layer for VGG-M; crashed in VOC07
    % and MIT Indoor for images which had one edge under 100px
    if min(crop_h,crop_w) * opts.scales(s) < max(border, 30), continue ; end
    %if sqrt(crop_h*crop_w) * opts.scales(s) > 1640, continue ; end
    if sqrt(crop_h*crop_w) * opts.scales(s) > 1024, continue ; end

    % resize the cropped image and extract features everywhere
    im_resized = imresize(im_cropped, opts.scales(s)) ;
    if opts.useSIFT
      [frames,descrs] = vl_dsift(mean(im_resized,3), ...
        'size', binSize, ...
        'step', stride, ...
        'fast', 'floatdescriptors') ;
      ur = unique(frames(1,:)) ;
      vr = unique(frames(2,:)) ;
      [u,v] = meshgrid(ur,vr) ;
      %assert(isequal([u(:)';v(:)'], frames)) ;
    else
      im_resized = bsxfun(@minus, im_resized, averageColour) ;
      if net.useGpu
        im_resized = gpuArray(im_resized) ;
      end
      res = vl_simplenn(net, im_resized, [], res, ...
        'conserveMemory', true, 'sync', true) ;
      w = size(res(end).x,2) ;
      h = size(res(end).x,1) ;
      descrs = permute(gather(res(end).x), [3 1 2]) ;
      descrs = reshape(descrs, size(descrs,1), []) ;
      % fixes padding / index out of bounds error.
      % TODO: needs checking.
      if offset < 0
          offset = offset + stride;
          w = w - 1;
          h = h - 1;
      end
      % seems a bit hacky way -- was w - 1; h - 1;
      % fixes index out of bounds error.
      [u,v] = meshgrid(...
        offset + (0:w-2) * stride, ...
        offset + (0:h-2) * stride) ;
    end

    u_ = (u - 1) / opts.scales(s) + 1 ;
    v_ = (v - 1) / opts.scales(s) + 1 ;
    loc_ = [u_(:)';v_(:)'] ;

    % for each region
    for r = 1:numel(regions_cropped.labels)
      mask_cropped = ismember(regions_cropped.basis, regions_cropped.labels{r}) ;
      mask_resized = imresize(mask_cropped, opts.scales(s), 'nearest') ;
      mask_features = mask_resized(sub2ind(size(mask_resized), v, u)) ;
      psi{r}{s} = descrs(:, mask_features) ;
      loc{r}{s} = loc_(:, mask_features) ;
      if 0
        figure(100) ; clf ;
        imagesc(vl_imsc(im_resized)) ; hold on ;
        plot(u,v,'g.') ;
        plot(u(mask_features),v(mask_features),'ro') ;
        axis equal ;
        drawnow ;
      end
    end
  end
  for r = 1:numel(psi)
    code{k}{r} = cat(2, psi{r}{:}) ;
    codeLoc{k}{r} = cat(2, zeros(2,0), loc{r}{:}) ;
    numReg = numReg + 1 ;
    numNull = numNull + isempty(code{k}{r}) ;
  end
end

if numNull > 0
  fprintf('%s: %d out of %d regions with null DCNN descriptor\n', ...
    mfilename, numNull, numReg) ;
end

% at this point code{i}{r} contains all local featrues for region r in
% image i
if isempty(opts.encoder)
  % no gmm: return the local descriptors, but not too many!
  rng(0) ;
  if (~isinf(opts.maxNumLocalDescriptorsReturned))
    for k=1:numel(code)
        for r = 1:numel(code{k})
            code{k}{r} = vl_colsubset(code{k}{r}, ...
                opts.maxNumLocalDescriptorsReturned) ;
        end
    end
  end
else
  numSelDescr = 250000;
  % encoding (supports BoVW, VLAD and FV)
  for k=1:numel(code)
    for r = 1:numel(code{k})
      descrs = opts.encoder.projection * bsxfun(@minus, code{k}{r}, ...
        opts.encoder.projectionCenter) ;
      if opts.encoder.renormalize
        descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
      end
      tmp = {} ;
      break_u = get_intervals(codeLoc{k}{r}(1,:), opts.numSpatialSubdivisions(1)) ;
      break_v = get_intervals(codeLoc{k}{r}(2,:), opts.numSpatialSubdivisions(2)) ;
      for spu = 1:opts.numSpatialSubdivisions(1)
        for spv = 1:opts.numSpatialSubdivisions(2)
          sel = ...
            break_u(spu) <= codeLoc{k}{r}(1,:) & codeLoc{k}{r}(1,:) < break_u(spu+1) & ...
            break_v(spv) <= codeLoc{k}{r}(2,:) & codeLoc{k}{r}(2,:) < break_v(spv+1);
          z = [];
          switch (opts.encoder.encoderType)
              case {'fv'}
                sel_descrs = descrs(:, sel);
                if (size(sel_descrs, 2) > numSelDescr)
                    sel_descrs = descrs(:, vl_colsubset(1: size(sel_descrs, 2), numSelDescr));
                end
                z = vl_fisher(sel_descrs, ...
                    opts.encoder.means, ...
                    opts.encoder.covariances, ...
                    opts.encoder.priors, ...
                    'Improved') ;
              case {'bovwsq'}
                [words, ~] = vl_kdtreequery(opts.encoder.kdtree, opts.encoder.words, ...
                                         descrs, 'MaxComparisons', 100) ;
                z = vl_binsum(zeros(opts.encoder.numWords,1), 1, double(words)) ;
                z = sign(z) .* sqrt(abs(z));
                z = bsxfun(@times, z, 1./max(1e-12, sqrt(sum(z .^ 2))));

              case {'bovw'}
                [words, ~] = vl_kdtreequery(opts.encoder.kdtree, opts.encoder.words, ...
                                         descrs, 'MaxComparisons', 100) ;
                z = vl_binsum(zeros(opts.encoder.numWords,1), 1, double(words)) ;
                z = bsxfun(@times, z, 1./max(1e-12, sqrt(sum(z .^ 2))));

              case {'vlad'}
                [words, ~] = vl_kdtreequery(opts.encoder.kdtree, opts.encoder.words, ...
                                         descrs, 'MaxComparisons', 15) ;
                assign = zeros(opts.encoder.numWords, numel(words), 'single') ;
                assign(sub2ind(size(assign), double(words), 1:numel(words))) = 1 ;
                z = vl_vlad(descrs, opts.encoder.words, assign, ...
                  'SquareRoot','NormalizeComponents') ;
              case {'llc'}
                [words, ~] = vl_kdtreequery(opts.encoder.kdtree, ...
                    single(opts.encoder.words), ...
                    single(descrs), 'MaxComparisons', 500, 'NumNeighbors', 5);
                z = LLCEncodeHelper(double(opts.encoder.words), ...
                    double(descrs), double(words), double(1e-4), false);
               % z = sign(z) .* sqrt(abs(z));
                z = bsxfun(@times, z, 1./max(1e-12, sqrt(sum(z .^ 2))));
          end
          tmp{end+1} = z;
        end
      end
      % normalization keeps norm = 1
      code{k}{r} = cat(1, tmp{:}) / (opts.numSpatialSubdivisions(1) * opts.numSpatialSubdivisions(2)) ;
    end
    code{k} = cat(2, code{k}{:}) ;
  end

  if nargout == 1
      clear codeLoc;
  end
end
% here code{i} is an array of FV descripors for each region, with one
% coulmn per region

function breaks = get_intervals(x,n)
if isempty(x)
  breaks = ones(1,n+1) ;
else
  x = sort(x(:)') ;
  breaks = x(round(linspace(1, numel(x), n+1))) ;
end
breaks(end) = +inf ;

% ------------------------------------------------------------------------
function [imCrop, regionsCrop] = crop(opts, im, regions, border)
% -------------------------------------------------------------------------

box = enclosingBox(regions.basis) ;

% include a border around it (feature support)
w = diff(box([1 3])) + border ;
h = diff(box([2 4])) + border ;
bx = mean(box([1 3])) ;
by = mean(box([2 4])) ;
sbox = round([bx - w/2 ; by - h/2 ; bx + w/2 ; by + h/2]) ;

% clip it
sbox = boxclip(sbox, [size(im,2), size(im,1)]) ;

% crop image and mask
sx = sbox(1):sbox(3) ;
sy = sbox(2):sbox(4) ;
imCrop = im(sy, sx, :) ;
regionsCrop = regions ;
regionsCrop.basis = regions.basis(sy, sx, :) ;

% -------------------------------------------------------------------------
function box = enclosingBox(mask)
% -------------------------------------------------------------------------
[x,y] = meshgrid(1:size(mask,2), 1:size(mask,1)) ;
x = x(any(mask,3)) ;
y = y(any(mask,3)) ;
box = [min(x) ; min(y) ; max(x) ; max(y)] ;
