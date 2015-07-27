function imdb = voc_get_database(vocDir, varargin)
opts.seed = 1 ;
opts = vl_argparse(opts, varargin) ;

assert(opts.seed == 1) ; % there is only one split

imdb.imageDir = fullfile(vocDir, 'JPEGImages') ;
imdb.maskDir = fullfile(vocDir, 'Masks'); % doesn't exist
imdb.classes.name={...
  'aeroplane'
  'bicycle'
  'bird'
  'boat'
  'bottle'
  'bus'
  'car'
  'cat'
  'chair'
  'cow'
  'diningtable'
  'dog'
  'horse'
  'motorbike'
  'person'
  'pottedplant'
  'sheep'
  'sofa'
  'train'
  'tvmonitor'};
numClass = length(imdb.classes.name);

% source images
imageFiles = dir(fullfile(imdb.imageDir, '*.jpg'));
imdb.images.name = {imageFiles.name};
numImages = length(imdb.images.name);

imdb.images.label = zeros(numClass, numImages);
imdb.images.label = zeros(1, numImages);
imdb.images.set = zeros(1, numImages);
imdb.images.id = 1:numImages;
imdb.images.vocid = cellfun(@(S) S(1:end-4), imdb.images.name, 'UniformOutput', false);
imageSets = {'train', 'val', 'test'};

% Loop over classes load labels
for c = 1:numClass,
  for s = 1:length(imageSets);
    imageSetPath = fullfile(vocDir, 'ImageSets', 'Main', sprintf('%s_%s.txt', imdb.classes.name{c}, imageSets{s}));
    [gtids,gt]=textread(imageSetPath,'%s %d');
    [membership, loc] = ismember(gtids, imdb.images.vocid);
    assert(all(membership));
    imdb.images.label(c, loc) = gt ;
  end
end

% Loop over images and record the imag sets
for s = 1:length(imageSets),
  imageSetPath = fullfile(vocDir, 'ImageSets', 'Main', sprintf('%s.txt',imageSets{s}));
  gtids = textread(imageSetPath,'%s');
  [membership, loc] = ismember(gtids, imdb.images.vocid);
  assert(all(membership));
  imdb.images.set(loc) = s;
end

% Remove images not part of train, val, test sets
valid = ismember(imdb.images.set, 1:length(imageSets));
imdb.images.name = imdb.images.name(imdb.images.id(valid));
imdb.images.id = 1:numel(imdb.images.name);
imdb.images.label = imdb.images.label(:, valid);
imdb.images.set = imdb.images.set(valid);
imdb.images.vocid = imdb.images.vocid(valid);

% Write out the segments
imdb.segments = imdb.images ;
imdb.segments.imageId = imdb.images.id ;
imdb.segments.difficult = false(1, numel(imdb.segments.id)) ;
% no masks

% make this compatible with the OS imdb
imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1,numel(imdb.meta.classes)) ;
