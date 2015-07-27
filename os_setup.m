function [opts, imdb] = os_setup(varargin)

global is_iasonas
if (exist('is_iasonas', 'var') ||isempty(is_iasonas) || ~is_iasonas)
    setup ;
end

opts.transf = {};
opts.seed = 1 ;
opts.batchSize = 128 ;
opts.useGpu = true ;
opts.regionBorder = 0.05 ;
opts.numDCNNWords = 64 ;
opts.numDSIFTWords = 256 ;
opts.numSamplesPerWord = 1000 ;
opts.printDatasetInfo = false ;
opts.excludeDifficult = true ;
opts.encoders = {struct('type', 'rcnn', 'opts', {})} ;
opts.dataset = 'os' ;
opts.osDir = 'data/os' ;
opts.fmdDir = 'data/fmd' ;
opts.dtdDir = 'data/dtd';
opts.kthDir = 'data/kth';
opts.alotDir = 'data/alot';
opts.mitDir = 'data/mit_indoor';
opts.msrcDir = 'data/msrc_c';
opts.cubDir = 'data/cub';
opts.vocDir = 'data/VOC2007';
opts.vocDir12 = 'data/VOC2012';
opts.writeResults = false;
opts.compid = 'comp2';
opts.publishDir = '~/Dropbox/Collaborations/Mircea Cimpoi/cvpr15/figures' ;
opts.suffix = 'baseline' ;
opts.prefix = 'v22' ;
opts.model = 'imagenet-vgg-m.mat';
opts.layer = 13 ; % for D-CNN (R-CNN is the penultimate)
opts.segProposalType = 'crisp' ;
opts.gpuId = 1;
opts.crf = [];
[opts, varargin] = vl_argparse(opts,varargin) ;

opts.expDir = sprintf('data/%s/%s-seed-%02d', opts.prefix, opts.dataset, opts.seed) ;
opts.imdbDir = fullfile(opts.expDir, 'imdb') ;
if ~iscell(opts.suffix)
  opts.resultPath = fullfile(opts.expDir, sprintf('result-%s.mat', opts.suffix)) ;
  opts.segResultPath = fullfile(opts.expDir, sprintf('%s/result-%s-seg.mat', opts.segProposalType, opts.suffix)) ;
  opts.segPublishDir = fullfile(opts.expDir, sprintf('%s/result-%s-seg-figures', opts.segProposalType, opts.suffix)) ;
  vl_xmkdir(opts.segPublishDir);
  if opts.writeResults
      opts.vocResultPath = fullfile(opts.expDir, [sprintf('result-%s/Main/%s_cls_', opts.suffix, opts.compid) '%s_%s.txt']);
  end
else
  for s = 1:numel(opts.suffix)
    opts.resultPath{s} = fullfile(opts.expDir, sprintf('result-%s.mat', opts.suffix{s})) ;
    opts.segResultPath{s} = fullfile(opts.expDir, sprintf('%s/result-%s-seg.mat', opts.segProposalType, opts.suffix{s})) ;
    opts.segPublishDir{s} = fullfile(opts.expDir, sprintf('%s/result-%s-seg-figures', opts.segProposalType, opts.suffix{s})) ;
    vl_xmkdir(opts.segPublishDir{s});
  end
end
opts = vl_argparse(opts,varargin) ;

if nargout <= 1, return ; end

% Setup GPU if needed
if opts.useGpu
  gpuDevice(opts.gpuId) ;
end

% -------------------------------------------------------------------------
%                                                            Setup encoders
% -------------------------------------------------------------------------

models = {} ;
for i = 1:numel(opts.encoders)
  if isstruct(opts.encoders{i})
    name = opts.encoders{i}.name ;
    opts.encoders{i}.path = fullfile(opts.expDir, [name '-encoder.mat']) ;
    opts.encoders{i}.codePath = fullfile(opts.expDir, [name '-codes.mat']) ;
    models = horzcat(models, get_cnn_model_from_encoder_opts(opts.encoders{i})) ;
  else
    for j = 1:numel(opts.encoders{i})
      name = opts.encoders{i}{j}.name ;
      opts.encoders{i}{j}.path = fullfile(opts.expDir, [name '-encoder.mat']) ;
      opts.encoders{i}{j}.codePath = fullfile(opts.expDir, [name '-codes.mat']) ;
      models = horzcat(models, get_cnn_model_from_encoder_opts(opts.encoders{i}{j})) ;
    end
  end
end

% -------------------------------------------------------------------------
%                                                       Download CNN models
% -------------------------------------------------------------------------

for i = 1:numel(models)
  if ~exist(fullfile('data/models', models{i}))
    fprintf('downloading model %s\n', models{i}) ;
    vl_xmkdir('data/models') ;
    urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models', models{i}),...
      fullfile('data/models', models{i})) ;
  end
end

% -------------------------------------------------------------------------
%                                                              Load dataset
% -------------------------------------------------------------------------

vl_xmkdir(opts.expDir) ;
vl_xmkdir(opts.imdbDir) ;

imdbPath = fullfile(opts.imdbDir, sprintf('imdb-seed-%d.mat', opts.seed)) ;
if exist(imdbPath)
  imdb = load(imdbPath) ;
  return ;
end

switch opts.dataset
  case 'os'
    imdb = os_get_database(opts.osDir) ;
  case 'os-a'
    imdb = os_attr_get_database(opts.osDir);
  case 'fmd'
    imdb = fmd_get_database(opts.fmdDir, 'seed', opts.seed) ;
  case 'dtd'
    imdb = dtd_get_database(opts.dtdDir, 'seed', opts.seed);
  case 'kth'
    imdb = kth_get_database(opts.kthDir, 'seed', opts.seed);
  case 'voc07'
    imdb = voc_get_database(opts.vocDir, 'seed', opts.seed);
  case 'voc12'
    imdb = voc_get_database(opts.vocDir12, 'seed', opts.seed);
  case 'voc12s'
    imdb = voc_get_seg_database(opts.vocDir12, 'seed', opts.seed);
  case 'mit'
    imdb = mit_indoor_get_database(opts.mitDir);
  case 'msrc'
    imdb = msrc_c_get_database(opts.msrcDir);
  case 'cubcrop'
    imdb = cub_get_database(opts.cubDir, true);
  case 'cub'
    imdb = cub_get_database(opts.cubDir, false);
  case 'alot'
    imdb = alot_get_database(opts.alotDir, 'seed', opts.seed);
  otherwise
    serror('Unknown dataset %s', opts.dataset) ;
end

save(imdbPath, '-struct', 'imdb') ;

if opts.printDatasetInfo
  print_dataset_info(imdb) ;
end

% -------------------------------------------------------------------------
function model = get_cnn_model_from_encoder_opts(encoder)
% -------------------------------------------------------------------------
p = find(strcmp('model', encoder.opts)) ;
if ~isempty(p)
  [~,m,e] = fileparts(encoder.opts{p+1}) ;
  model = {[m e]} ;
else
  model = {} ;
end
