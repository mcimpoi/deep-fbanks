function imdb = os_get_database(osDir)

load(fullfile(osDir, 'imdb/imdb.mat'), 'imdb') ;
imdb.imageDir512 = fullfile(osDir, imdb.imageDir512) ;
imdb.imageDir1024 = fullfile(osDir, imdb.imageDir1024) ;
imdb.maskDir512 = fullfile(osDir, imdb.maskDir512) ;
imdb.maskDir1024 = fullfile(osDir, imdb.maskDir1024) ;

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

% finally, merge the background classes
% bkg = [18 25]
imdb.segments.label(imdb.segments.label == 25) = 18 ;
imdb.meta.inUse(25) = false ;
imdb.meta.classes{18} = 'other' ;
imdb.meta.classes{25} = 'other' ;

% no difficult regions by default
imdb.segments.difficult = false(1, numel(imdb.segments.id)) ;
