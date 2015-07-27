function run_experiments_seg_summary()
rcnn.name = 'rcnn' ;
rcnn.opts = {...
  'type', 'rcnn', ...
  'model', 'data/models/imagenet-vgg-m.mat', ...
  'layer', 19} ;

rcnnvd.name = 'rcnnvd' ;
rcnnvd.opts = {...
  'type', 'rcnn', ...
  'model', 'data/models/imagenet-vgg-verydeep-19.mat', ...
  'layer', 41} ;

dcnn.name = 'dcnn' ;
dcnn.opts = {...
  'type', 'dcnn', ...
  'model', 'data/models/imagenet-vgg-m.mat', ...
  'layer', 13, ...
  'numWords', 64} ;

dcnnvd.name = 'dcnnvd' ;
dcnnvd.opts = {...
  'type', 'dcnn', ...
  'model', 'data/models/imagenet-vgg-verydeep-19.mat', ...
  'layer', 35, ...
  'numWords', 64} ;

dsift.name = 'dsift' ;
dsift.opts = {...
  'type', 'dsift', ...
  'numWords', 256, ...
  'numPcaDimensions', 80} ;

datasetList = { 'msrc'} ; % add 'os'
numSplits = [1, 1] ;

setupNameList = {'rcnn', 'dcnn', 'rdcnn'} ;
encoderList = {{rcnn}, {dcnn}, {rcnn dcnn}} ;
if 1
  setupNameList =  horzcat(setupNameList, {'rcnnvd', 'dcnnvd', 'rdcnnvd'}) ;
  encoderList = horzcat(encoderList, {{rcnnvd}, {dcnnvd}, {rcnnvd dcnnvd}}) ;
end

t = LatexTable() ;
t.begin().hline().vline().pf('dataset').pf('measure (\\%%)').vline() ;
for ee = 1: numel(setupNameList)
  t.pf('%s', upper(setupNameList{ee})) ;
end
t.vline().pf('SoA') ;
t.vline().endl().hline() ;

for ii = 1 : numel(datasetList)
  switch datasetList{ii}
    case 'msrc'
      dn = 'MSRC' ; m = 'msrcAcc' ;
      art = '86.5~\cite{ladicky10what}' ;
    case 'os'
      dn = 'OS' ; m = 'acc' ;
      art = '--' ;
    otherwise, assert(false) ;
  end
  t.pf(dn).pf(m) ;
  for ee = 1: numel(encoderList)
    score = [] ;
    for jj = 1 : numSplits(ii)
      opts =  os_setup(...
        'dataset', datasetList{ii}, 'seed', jj, ...
        'encoders', encoderList{ee}, ...
        'prefix', 'exp01', ...
        'suffix', setupNameList{ee}) ;
      fprintf('loading %s\n', opts.segResultPath) ;
      res = load(opts.segResultPath) ;
      switch m
        case 'map11', score(end+1) = res.test.map11 ;
        case 'msrcAcc', score(end+1) = res.test.msrcAcc ;
        case 'acc', score(end+1) = res.test.acc ;
        otherwise, assert(false) ;
      end
    end
    if numel(score) > 1
      t.pf('%5.1f \\tiny$\\pm%3.1f$', 100*mean(score), 100*std(score)) ;
    else
      t.pf('%5.1f', 100*mean(score)) ;
    end
    if jj == 1
      figure(1) ; clf ;
      switch m
        case 'acc'
          imagesc(res.test.confusion) ;
          title(sprintf('%s accuracy: %.1f %%', upper(opts.suffix), res.test.acc*100)) ;
          vl_xmkdir(opts.segPublishDir) ;
          vl_figaspect(1) ;
          vl_printsize(.7) ;
          print(fullfile(opts.segPublishDir, ['confusion.pdf']), '-dpdf') ;
      end
    end
  end
  t.pf('%s',art).endl() ;
end
t.hline() ;
str = t.end() ;
disp(str) ;
