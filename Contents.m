% SaddlepointExpansions for MATLAB: saddlepoint expansions of any order from the cumulant
% generating function (Broda and Kan); MATLAB counterpart of SaddlepointExpansions.jl.
%
% Cumulant generating functions and expansions
%   CGF                   - a cumulant generating function; methods tailprob, density,
%                           ratio_density, ratio_tailprob, saddlepoint, derivatives, taylor,
%                           reflect, marginal, linear_combination, sample_mean, shift, plus
%   tps                   - truncated multivariate power series (Taylor-mode differentiation)
%   spx_setup             - adds the package and its engine to the path
%
% Library
%   normal_cgf            - multivariate normal
%   gamma_cgf             - gamma (and exponential)
%   ncx2_cgf              - noncentral chi-squared
%   exponential_sums_cgf  - linear combinations of independent exponentials, sample means
%   quadratic_forms_cgf   - quadratic forms in a normal vector (closed-form derivatives)
%   wishart_diag_cgf      - diagonal of a Wishart matrix
%   common_component_cgf  - noncentral chi-squared variables with a common component
%   uniform_cgf           - uniform distribution
%   sinhc, sincu          - sinh(x)/x and sin(x)/x, also for series
%
% Engine (folder engine): spa_saddle, spa_uni, spa_bi, spa_orthant2, spa_orthant, spa_mv, spa_tri,
%   spa_ratio_pdf, qf_Kmulti, ncx2_spa, ncf_spa, dncf_spa, ... (derivative interface Kfun(nu, t))
%
% Tests, examples and manual: tests/test_spx, examples/demo_spx, doc/manual.pdf
