function imdb = os_get_database(osDir)

load(fullfile(osDir, 'imdb/imdb.mat'), 'imdb') ;
imdb.imageDir512 = fullfile(osDir, imdb.imageDir512) ;
imdb.imageDir1024 = fullfile(osDir, imdb.imageDir1024) ;
imdb.maskDir512 = fullfile(osDir, imdb.maskDir512) ;
imdb.maskDir1024 = fullfile(osDir, imdb.maskDir1024) ;

opts.seed = 1;

% To avoid classes which are not labelled yet
% or underrepresented; banded has 87
minPerClass = 80;

cls_names = dir('data/os/labels/*.txt');
cls_names = {cls_names.name};

imdb.classes.name = {};
segIds = {};
labels = {};

for ii = 1 : numel(cls_names)
  fid = fopen(fullfile('data/os/labels/', cls_names{ii}));
  if (fid > 0)
    lines = textscan(fid, '%s%d%d%d');
    gt_labels = lines{4};
    if (sum(gt_labels == 1) >= minPerClass)
      imdb.classes.name{end + 1} = cls_names{ii};
    else
      continue;
    end
    segIds{end + 1} = lines{3};
    labels{end + 1} = lines{4};
  end
end



% use these by default
imdb.imageDir = imdb.imageDir512 ;
imdb.maskDir = imdb.maskDir512 ;
imdb.segmDir = fullfile(osDir, 'segm/512') ;

% split images in train, val, test
n = numel(imdb.images.id) ;
m = round(n/3) ;
sets = [1 * ones(1,m), 2 * ones(1,m), 3 * ones(1,n-2*m)] ;
rng(0) ;
imdb.images.set = sets(randperm(n)) ;
[~,i] = ismember(imdb.segments.imageId, imdb.images.id) ;
imdb.segments.set = imdb.images.set(i) ;
imdb.segments.label = imdb.segments.materialClass ;

% now remove all the segments that belong to classes that are not in use
ok = logical(imdb.meta.inUse(imdb.segments.label)) ;
imdb.segments = soaSubsRef(imdb.segments, ok) ;

imdb.segments.label = zeros(numel(imdb.classes.name), ...
  numel(imdb.segments.id));

for ii = 1 : numel(imdb.classes.name)
  % don't understand why this doesn't work
  %[lia, ~] = ismember(imdb.segments.id, segIds{ii});
  %imdb.segments.label(ii, lia) = labels{ii}';

  pos_segments = segIds{ii}(labels{ii} == 1);
  neg_segments = segIds{ii}(labels{ii} == -1);

  [lia, ~] = ismember(imdb.segments.id, pos_segments);
  imdb.segments.label(ii, lia) = 1;
  [lia, ~] = ismember(imdb.segments.id, neg_segments);
  imdb.segments.label(ii, lia) = -1;
end


% finally, merge the background classes
% bkg = [18 25]
%imdb.segments.label(imdb.segments.label == 25) = 18 ;
%imdb.meta.inUse(25) = false ;
%imdb.meta.classes{18} = 'other' ;
%imdb.meta.classes{25} = 'other' ;

imdb.meta.classes = imdb.classes.name;
imdb.meta.inUse = ones(1, numel(imdb.classes.name));

% no difficult regions by default
imdb.segments.difficult = false(1, numel(imdb.segments.id)) ;
save(fullfile(osDir, 'imdb/imdb_attr.mat'), 'imdb') ;
