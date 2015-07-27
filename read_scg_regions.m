function regions = read_scg_regions(filePath, varargin)
opts.maxNumRegions = +inf ;
opts = vl_argparse(opts, varargin) ;

tmp = load(filePath) ;
regions.basis  = tmp.candidates_scg.superpixels ;
regions.labels = tmp.candidates_scg.labels(:)' ;
regions.scores = tmp.candidates_scg.scores(:)' ;

sareas = accumarray(regions.basis(:), 1) ;
for i=1:numel(regions.labels)
  regions.areas(i) = sum(sareas(regions.labels{i})) ;
end

% take only the top regions
[~,perm] = sort(regions.scores, 'descend') ;
perm = vl_colsubset(perm, opts.maxNumRegions, 'beginning') ;
regions.labels = regions.labels(perm) ;
regions.scores = regions.scores(perm) ;
regions.areas = regions.areas(perm) ;
