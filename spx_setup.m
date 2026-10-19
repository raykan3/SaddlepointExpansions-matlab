function spx_setup
%SPX_SETUP Adds the SaddlepointExpansions package and its engine to the MATLAB path.
here = fileparts(mfilename('fullpath'));
addpath(here, fullfile(here, 'engine'));
end
