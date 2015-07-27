classdef LatexTable < handle
  properties
    rowSuffixes
    rowPrefixes
    colPrefixes
    entries
    r
    c
  end

  methods
    function obj = begin(obj)
      obj.rowSuffixes = {} ;
      obj.rowPrefixes = {''} ;
      obj.colPrefixes = {''} ;
      obj.entries = {} ;
      obj.r = 1 ;
      obj.c = 1 ;
    end
    function obj = pf(obj, str, varargin)
      obj.entries{obj.r,obj.c} = sprintf(str, varargin{:}) ;
      obj.c = obj.c + 1 ;
      if obj.r == 1, obj.colPrefixes{obj.c} = '' ; end
    end
    function obj = endl(obj)
      obj.rowSuffixes{obj.r} = sprintf('\\\\\n') ;
      obj.c = 1 ;
      obj.r = obj.r + 1 ;
      obj.rowPrefixes{obj.r} = '' ;
    end
    function obj = vline(obj)
      obj.colPrefixes{obj.c} = horzcat(obj.colPrefixes{obj.c}, '|') ;
    end
    function obj = hline(obj)
      obj.rowPrefixes{obj.r} = horzcat(obj.rowPrefixes{obj.r}, sprintf('\\hline\n')) ;
    end
    function move(obj, r, c)
      obj.r = r ;
      obj.c = c ;
    end
    function str = end(obj)
      str = {} ;
      nc = size(obj.entries,2) ;
      nr = size(obj.entries,1) ;
      sizes = cellfun(@(x) numel(x), obj.entries) ;
      widths = max(sizes,[],1) ;

      str{end+1} = '\begin{tabular}{' ;
      for c=1:nc
        str{end+1} = [obj.colPrefixes{c} 'c'] ;
      end
      str{end+1} = sprintf('%s}\n', obj.colPrefixes{nc+1}) ;

      for r = 1:nr
        str{end+1} = obj.rowPrefixes{r} ;
        for c = 1:nc
          format = sprintf('%%%ds', widths(c)) ;
          if c > 1, format = [' &' format] ; end
          str{end+1} = sprintf(format, obj.entries{r,c}) ;
        end
        str{end+1} = obj.rowSuffixes{r} ;
      end
      str{end+1} = obj.rowPrefixes{nr+1} ;
      str{end+1} = sprintf('\\end{tabular}\n') ;
      str = horzcat(str{:}) ;
    end
  end
end