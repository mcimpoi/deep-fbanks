function print_dataset_info(imdb)
multiLabel = (size(imdb.segments.label, 1) > 1) ;
train = ismember(imdb.images.set, [1 2]) ;
test = ismember(imdb.images.set, [3]) ;
fprintf('dataset: classes: %d in use. These are:\n', sum(imdb.meta.inUse)) ;
trainSeg = ismember(imdb.segments.imageId, imdb.images.id(train)) ;
testSeg = ismember(imdb.segments.imageId, imdb.images.id(test)) ;
for i = find(imdb.meta.inUse)
  if ~multiLabel
    a = sum(imdb.segments.label(trainSeg) == i) ;
    b = sum(imdb.segments.label(testSeg) == i) ;
    c = sum(imdb.segments.label == i) ;
  else
    a = sum(imdb.segments.label(i, trainSeg) > 0, 2) ;
    b = sum(imdb.segments.label(i, testSeg) > 0, 2) ;
    c = sum(imdb.segments.label(i, :) > 0, 2) ;
  end
  fprintf('%4d: %15s (train: %5d, test: %5d total: %5d)\n', ...
    i, imdb.meta.classes{i}, a, b, c) ;
end
a = numel(trainSeg) ;
b = numel(testSeg) ;
c = numel(imdb.segments.id) ;
fprintf('%4d: %15s (train: %5d, test: %5d total: %5d)\n', ...
  +inf, '**total**', a, b, c) ;
fprintf('dataset: there are %d images (%d trainval %d test)\n', ...
  numel(imdb.images.id), sum(train), sum(test)) ;