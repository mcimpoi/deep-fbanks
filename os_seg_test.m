function os_seg_test(varargin)
% note: this function runs multiple setups (feature combinations) at the
% same time

% load stuff
[opts, imdb] = os_setup(varargin{:}) ;
classes = find(imdb.meta.inUse) ;

figure(200) ; clf ;
cmap = plot_legend(imdb) ;
for s = 1:numel(opts.suffix)
  vl_xmkdir(opts.segPublishDir{s});
  print(fullfile(opts.segPublishDir{s}, [opts.dataset '-legend.pdf']), '-dpdf') ;
end

info = struct() ;
encoders = struct() ;
models = struct() ;
for s = 1:numel(opts.suffix)
  vl_xmkdir(fileparts(opts.segResultPath{s})) ;
  setupName = opts.suffix{s} ;
  fprintf('preparing setup %s\n', setupName) ;

  % make a directory to contain the segment scores for each image
  [setupLoc,setupBaseName,~] = fileparts(opts.segResultPath{s}) ;
  vl_xmkdir(fullfile(setupLoc, setupBaseName)) ;

  % source all encoders for this setup (could be the same encoder of
  % another setup)
  for j = 1:numel(opts.encoders{s})
    name = opts.encoders{s}{j}.name ;
    if ~isfield(encoders, 'name')
      fprintf('loading encoder %s\n', name) ;
      encoders.(name) = load(opts.encoders{s}{j}.path) ;
      if isfield(encoders.(name), 'net')
        if opts.useGpu
          encoders.(name).net = vl_simplenn_move(encoders.(name).net, 'gpu') ;
          encoders.(name).net.useGpu = true ;
        else
          encoders.(name).net = vl_simplenn_move(encoders.(name).net, 'cpu') ;
          encoders.(name).net.useGpu = false ;
        end
      end
    end
  end

  % load the SVM classifiers for this setup
  tmp = load(opts.resultPath{s}) ;
  models(s).w = tmp.w ;
  models(s).b = tmp.b ;

  info.(setupName) = init_info(classes) ;
end

% -------------------------------------------------------------------------
%                                                            Process images
% -------------------------------------------------------------------------

rng(0) ;
printThisImage = false(1, numel(imdb.images.id)) ;
test = find(imdb.images.set <= 3) ;
printThisImage(vl_colsubset(test, 60)) = true ;
confusion = cell(1, numel(imdb.images.id)) ;

%fprintf('num labs: %d\n', matlabpool('size')) ;
parfor i = 1:numel(imdb.images.id)
  confusion{i} = do_one_image(opts, encoders, models, imdb, classes, cmap, printThisImage, i) ;
end

for s = 1:numel(opts.suffix)
  setupName = opts.suffix{s} ;
  for i = 1:numel(imdb.images.id)
    if ismember(imdb.images.set(i), [1 2]), setName = 'train' ; else setName = 'test' ; end
    info.(setupName).(setName).confusion_ = info.(setupName).(setName).confusion_ + confusion{i}{s} ;
  end
  for setName = {'train', 'test'}
    setName = char(setName) ;
    [info.(setupName).(setName).confusion, ...
      info.(setupName).(setName).acc, ...
      info.(setupName).(setName).msrcAcc] = ...
      nconfusion(info.(setupName).(setName).confusion_) ;
  end

  % save
  [setupLoc,setupBaseName,~] = fileparts(opts.segResultPath{s}) ;
  info_ = info.(setupName) ;
  save(opts.segResultPath{s}, '-struct', 'info_') ;

  % print the confusion
  figure(1) ; clf ; t = 0 ;
  for setName = {'train', 'test'}
    t = t + 1 ;
    setName = char(setName) ;
    subplot(1,2,t) ; imagesc(info.(setupName).(setName).confusion) ; axis equal ;
    title(sprintf('%s %s acc:%.2f%% (msrc acc: %.2f%%)', ...
      setupName, setName, ...
      info.(setupName).(setName).acc*100, ...
      info.(setupName).(setName).msrcAcc*100)) ;
  end
  vl_printsize(1) ;
  print(fullfile(setupLoc, [setupBaseName '.pdf']), '-dpdf') ;
end

function saveIntermediate(scoresPath, scores, confusion_)
save(scoresPath, 'scores', 'confusion_') ;

function confusion_ = loadIntermediate(scoresPath)
load(scoresPath, 'confusion_') ;

% -------------------------------------------------------------------------
function confusion = do_one_image(opts, encoders, models, imdb, classes, cmap, printThisImage, i)
% -------------------------------------------------------------------------
if ismember(imdb.images.set(i), [1 2]), setName = 'train' ; else setName = 'test' ; end
task = getCurrentTask() ;
if ~isempty(task), tid = task.ID ; else tid = 1 ; end
fprintf('Task %5d: segmenting image %d of %d (%s)\n', tid, i, numel(imdb.images.id), setName) ;
[~,baseName,~] = fileparts(imdb.images.name{i}) ;
features = struct() ;
for s = 1:numel(opts.suffix)
  setupName = opts.suffix{s} ;
  [setupLoc,setupBaseName,~] = fileparts(opts.segResultPath{s}) ;
  scoresPath = fullfile(setupLoc, setupBaseName, [baseName '-scores.mat']) ;
  nClasses = numel(classes);


  seg_labels = imdb.segments.label(:, imdb.segments.imageId == imdb.images.id(i));
  if (max(max(seg_labels)) == 0)
    confusion_ = zeros(nClasses, nClasses + 1);
  elseif exist(scoresPath)
    % load previous results
    confusion_ = loadIntermediate(scoresPath) ;
  else
    % get the region proposals
    im = imread(fullfile(imdb.imageDir, imdb.images.name{i})) ;
    switch opts.segProposalType
      case 'crisp'
        regions = read_crisp_regions(...
          fullfile(imdb.segmDir, 'crisp', sprintf('%s.png', baseName)), ...
          [size(im,1) size(im,2)]) ;
      case 'scg'
        regions = read_scg_regions(...
          fullfile(imdb.segmDir, 'scg_proposals', sprintf('%s.mat', baseName))) ;
      case 'scg200'
        regions = read_scg_regions(...
          fullfile(imdb.segmDir, 'scg_proposals', sprintf('%s.mat', baseName)), ...
          'maxNumRegions', 200) ;
    end

    % get the region features
    psi = {} ;
    for j = 1:numel(opts.encoders{s})
      n = opts.encoders{s}{j}.name ;
      if ~isfield(features, n)
        features.(n) = encoder_extract_for_regions(encoders.(n), im, regions) ;
      end
      psi{j} = features.(n) ;
    end
    psi = cat(1, psi{:}) ;

    % evaluate classifiers
    scores = bsxfun(@plus, models(s).w' * psi, models(s).b') ;

    % compute confusion and save results
    [~,~,gt] = os_get_gt_regions(imdb, imdb.images.id(i)) ;
    [confusion_, gt_, pred_] = evaluate_seg(classes, gt, regions, scores) ;
    saveIntermediate(scoresPath, scores, confusion_) ;

    % optionally print the image
    if printThisImage(i)
      err = (gt_ ~= pred_) & (gt_ > 0) ;
      good = (gt_ == pred_) & (gt_ > 0) ;
      name = sprintf('%s-%s', baseName, setupName) ;
      pdir = opts.segPublishDir{s} ;
      imwrite(im, fullfile(pdir, [baseName '.jpg'])) ;
      imwrite(gt_+1, colormap(cmap), fullfile(pdir, [name '-gt.png'])) ;
      imwrite(pred_+1, colormap(cmap), fullfile(pdir, [name '-pred.png'])) ;
      imwrite(1 + err + 2*good, [[1 1 1];[1 0 0];[0 1 0]], fullfile(pdir, [name '-err.png'])) ;
    end
  end
  confusion{s} = confusion_ ;
end

% -------------------------------------------------------------------------
function [c, gt_, pred_] = evaluate_seg(classes, gt, regions, scores)
% -------------------------------------------------------------------------
numClasses = numel(classes) ;
if (numel(size(gt)) == 2)
    pred = zeros(size(gt)) ;
else
    pred = zeros(size(gt, 2), size(gt, 3));
end
%scores = scores(1:23,:) ;
[~, labels] = max(scores,[],1) ;
for r=1:numel(labels)
  mask = ismember(regions.basis, regions.labels{r}) ;
  pred(mask) = classes(labels(r)) ;
end

if (numel(size(gt)) == 2)
sel = find(gt(:) > 0) ;
[~, gt_] = ismember(gt, classes) ;
[~, pred_] = ismember(pred, classes) ;
pred_(pred_ == 0) = numClasses + 1 ;
c = accumarray([gt_(sel), pred_(sel)], 1, [numClasses numClasses+1]) ;
else

    [~, pred_] = ismember(pred, classes) ;
    pred_(pred_ == 0) = numClasses + 1 ;
    gt_ = zeros(size(gt, 2), size(gt, 3));
    % fill all labels; we do not care of overlaps for now;
    for ll = 1 : size(gt, 1)
        gt0 = reshape(gt(ll, :, :), size(gt, 2), size(gt, 3));
        gt_(gt0 == 1) = ll;
    end

    % now make sure ground truth matches the predictions
    for ll = 1 : size(gt, 1)
        gt0 = reshape(gt(ll, :, :), size(gt, 2), size(gt, 3));
        sel_gt = gt0 == 1;
        sel_pred = pred_ == ll;
        mask_ = min(sel_pred, sel_gt);
        gt_(mask_) = ll;
    end
    sel = find(gt_(:) > 0);
    c = accumarray([gt_(sel), pred_(sel)], 1, [numClasses numClasses+1]) ;
end
if 0
  cmap = [[1 1 1] ; distinguishable_colors(numel(classes))] ;
  figure(100) ; clf ;
  subplot(1,2,1); image(gt_+1); axis equal ; title('(partial) gt') ;
  subplot(1,2,2); image(pred_+1); axis equal ; title('predicted') ;
  colormap(cmap) ;
  drawnow ;
  keyboard
end

% -------------------------------------------------------------------------
function [c,acc,msrcAcc]=nconfusion(c)
% -------------------------------------------------------------------------
msrcAcc = sum(diag(c)) / sum(c(:)) ;
c = bsxfun(@times, 1./max(sum(c,2),1e-10), c) ;
acc = mean(diag(c)) ;

% -------------------------------------------------------------------------
function info = init_info(classes)
% -------------------------------------------------------------------------
nc = numel(classes) ;
info.train.confusion_ = zeros(nc,nc+1) ;
info.train.confusion = zeros(nc,nc+1) ;
info.test.confusion_ = zeros(nc,nc+1) ;
info.test.confusion = zeros(nc,nc+1) ;
info.train.acc = 0 ;
info.train.msrcAcc = 0 ;
info.test.acc = 0 ;
info.test.msrcAcc = 0 ;
info.lastProcessedImage = 0 ;

% -------------------------------------------------------------------------
function cmap = plot_legend(imdb)
% -------------------------------------------------------------------------
numClasses = sum(imdb.meta.inUse) ;
classes = find(imdb.meta.inUse) ;
cmap = [[1 1 1] ; distinguishable_colors(numClasses) ;] ;
h = 16 ;
set(gca,'units', 'points', 'position',[0 0 100*8 h*3]) ;
set(gcf,'paperunits', 'points', 'paperposition',[0 0 150*8 h*3]) ;
set(gcf,'paperunits', 'points', 'papersize',[100*8 h*3]) ;

for i=1:3
  for j=1:8
    t = (i-1)*8 + j;
    if t > numClasses, break ; end
    x = (j-1)*100 ;
    y = (4-i-1)*h ;
    text(x+1.5*h, y+0.5*h, imdb.meta.classes{classes(t)}, ...
      'interpreter', 'none', 'verticalalign', 'middle', 'fontsize', 10) ;
    patch([0 h h 0]+x,[0 0 h h]+y,cmap(t+1,:)) ;
  end
end
axis equal off ;
xlim([0 100*8])
ylim([0 h*3]) ;

% -------------------------------------------------------------------------
function code = encoder_extract_for_regions(encoder, im, regions)
% -------------------------------------------------------------------------
switch encoder.type
  case 'rcnn'
    code = get_rcnn_features(encoder.net, ...
      im, regions, ...
      'regionBorder', encoder.regionBorder) ;
  case 'dcnn'
    code = get_dcnn_features(encoder.net, ...
      im, regions, ...
      'encoder', encoder) ;
  case 'dsift'
    code = get_dcnn_features([], im, regions, ...
      'useSIFT', true, ...
      'encoder', encoder) ;
end
code = cat(2, code{:}) ;
