classdef testSparseBLASMexAPIs < matlab.unittest.TestCase
% testSparseBLASMexAPIs Tests for SparseBLAS MEX APIs

   methods(TestMethodSetup)
        function initializeRNG(~)
            rng(0,'twister');
        end
   end

   properties(TestParameter)
        % Loop over different sizes
        sizesToTest = struct(...
            'empty1',    0, ...
            'empty2',    1, ...
            'tiny1',     2, ...
            'tiny2',     5, ...
            'small1' , 1e1, ...
            'small2',  5e1, ...
            'medium1', 1e2, ...
            'medium2', 5e2, ...
            'large1',  1e3, ...
            'large2',  5e3);
        % Loop over data types and complexities
        complexity = {'real', 'complex'};
        datatypes  = {'double', 'single'};
        % Loop over various shapes of sparse and dense inputs
        shape      = {'square', 'tall', 'wide'};
        numberOfRHS = struct(...
            'singleColumn',      1, ...
            'doubleColumn',      2, ...
            'manyColumns',      10, ...
            'veryManyColumns', 100);
        % Loop over different scalar scalings
        alpha      = struct(...
            'none',          [],   ...
            'neutral',       1.0,  ...
            'upScale',       2.3,  ...
            'upScaleNeg',   -4.3,  ...
            'downScale',     0.23, ...
            'downScaleNeg', -0.43);
    end

    methods (Test)
        % Test each API
        function simpleSPMV(testCase, sizesToTest, datatypes, complexity, shape, alpha)
            %% Create data
            nRhs = 1;
            [A, x] = createData(sizesToTest, nRhs, datatypes, complexity, shape);

            %% Calculate reference solution, adapted to MATLAB's special
            %  case treatment
            if isempty(alpha)
                y_exp = A*x;
            else
                alpha = cast(alpha, datatypes);
                if strcmp(complexity, 'complex')
                    alpha = complex(alpha, alpha);
                end
                y_exp = alpha*A*x;
            end

            % If either input the '*' is scalar, MATLAB calls '.*' which
            % returns sparse results for dense and sparse mixed inputs,
            % hence, making the result dense as SparseBLAS doesn't special
            % case these situations.
            if isscalar(A) || isscalar(x)
                y_exp = full(y_exp);
            end

            % MATLAB strips all-zero imaginary parts during '*'. SparseBLAS
            % does not, hence, make results complex again if complexity is
            % set to 'complex'.
            if strcmp(complexity, 'complex') && isreal(y_exp)
                y_exp = complex(y_exp);
            end

            %% Calculate solution via SparseBLAS MEX APIs
            if isempty(alpha)
                y_act = simple_spmv_mex(A, x);
            else
                y_act = simple_spmv_mex(A, x, alpha);
            end

            %% Verify results
            testCase.verifyEqual(y_act, y_exp);
        end

        function simpleSPMM(testCase, sizesToTest, numberOfRHS, datatypes, complexity, shape, alpha)
            %% Create data
            [A, X] = createData(sizesToTest, numberOfRHS, datatypes, complexity, shape);

            %% Calculate reference solution, adapted to MATLAB's special
            %  case treatment
            if isempty(alpha)
                y_exp = A*X;
            else
                alpha = cast(alpha, datatypes);
                if strcmp(complexity, 'complex')
                    alpha = complex(alpha, alpha);
                end
                y_exp = alpha*A*X;
            end

            % If either input the '*' is scalar, MATLAB calls '.*' which
            % returns sparse results for dense and sparse mixed inputs,
            % hence, making the result dense as SparseBLAS doesn't special
            % case these situations.
            if isscalar(A) || isscalar(X)
                y_exp = full(y_exp);
            end

            % MATLAB strips all-zero imaginary parts during '*'. SparseBLAS
            % does not, hence, make results complex again if complexity is
            % set to 'complex'.
            if strcmp(complexity, 'complex') && isreal(y_exp)
                y_exp = complex(y_exp);
            end

            %% Calculate solution via SparseBLAS MEX APIs
            if isempty(alpha)
                y_act = simple_spmm_mex(A, X);
            else
                y_act = simple_spmm_mex(A, X, alpha);
            end

            %% Verify results
            testCase.verifyEqual(y_act, y_exp);
        end
    end

end

function [A, X] = createData(n, nRhs, datatypes, complexity, shape)
% We use this routine to create sparse A and dense X
lesser_n = floor(n/2);
switch shape
    case 'wide'
        sz = [lesser_n, n];
    case 'tall'
        sz = [n, lesser_n];
    case 'square'
        sz = [n, n];
end

switch complexity
    case 'real'
        A = sprand(sz(1), sz(2), 0.01, datatypes);
        X = rand(sz(2), nRhs, datatypes);
    case 'complex'
        A = complex(sprand(sz(1), sz(2), 0.01, datatypes), ...
            sprand(sz(1), sz(2), 0.01, datatypes));
        X = complex(rand(sz(2), nRhs, datatypes), ...
            rand(sz(2), nRhs, datatypes));
end
end
