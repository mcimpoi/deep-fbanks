function imdb = kth_get_database(kthDir, varargin)
opts.seed = 1;
opts = vl_argparse(opts, varargin) ;

rng(opts.seed) ;

imdb.imageDir = fullfile(kthDir) ;
imdb.maskDir = fullfile(kthDir, 'mask') ;

if (~exist(kthDir, 'dir'))
    error('Dataset missing! Make sure you have the dataset downloaded in data folder');
end

cats = dir(imdb.imageDir) ;
cats = cats([cats.isdir] & ~ismember({cats.name}, {'.','..'})) ;
imdb.classes.name = {cats.name} ;
imdb.images.id = [] ;

for c=1:numel(cats)
  ims = [];
  imNames = {};
  for ss = 'a' : 'd'
    tmp = dir(fullfile(imdb.imageDir, imdb.classes.name{c}, ...
      ['sample_' ss], '*.png'));
    ims = cat(1, ims, tmp);
    imNames = [imNames, cellfun(@(S) strcat('sample_', ss, '/', S), ...
      {tmp.name}, 'Uniform', 0)];
  end
  %imdb.images.name{c} = fullfile(imdb.classes.name{c}, {ims.name}) ;
  imdb.images.name{c} = cellfun(@(S) fullfile(imdb.classes.name{c}, S), ...
    imNames, 'Uniform', 0);
  imdb.images.label{c} = c * ones(1,numel(ims)) ;
  if numel(ims) ~= 432, error('KTH data inconsistent') ; end
  sampleId = [1 * ones(1, 108), 2 * ones(1, 108), ...
    3 * ones(1, 108), 4 * ones(1, 108)];
  imdb.images.set{c} = 3 * (sampleId ~= opts.seed) + ...
    1 * (sampleId == opts.seed);
end
imdb.images.name = horzcat(imdb.images.name{:}) ;
imdb.images.label = horzcat(imdb.images.label{:}) ;
imdb.images.set = horzcat(imdb.images.set{:}) ;
imdb.images.id = 1:numel(imdb.images.name) ;

imdb.segments = imdb.images ;
imdb.segments.imageId = imdb.images.id ;
% there are no segment masks

% make this compatible with the OS imdb
imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1, numel(imdb.meta.classes)) ;
imdb.segments.difficult = false(1, numel(imdb.segments.id)) ;
