%% Basic Operations-------------------------------------------------------------
PS1('> ') % change prompt

A = [1 2; 3 4; 5 6];
disp(A)

v = [1 2 3]; % row vector
disp(v);

v2 = [1; 2; 3]; % col vec
disp(v2);

v3 = 1:0.1:2; % R: seq(1, 2, 0.1)
disp(v3);

v4 = 1:7;
disp(v4)

ones23 = ones(2, 3);
disp(ones23);

zeros22 = zeros(2, 2);
disp(zeros22);

r = rand(4, 3);
disp(r);

rnorm = randn(2, 2);
disp(rnorm);

w = -6 + sqrt(10) * randn(1, 10000);
hist(w);
hist(w, 50); % 50 bins

I5 = eye(5); % identity matrix
disp(I5);

% help x % help for x;



%% Moving Data Around-----------------------------------------------------------
size(A)    % Py: A.shape
size(A, 1) %     A.shape[0]
size(A, 2)

v = [1 2 3 4];
length(v) % 4

disp(pwd)
disp(ls)

load ./machine-learning-ex1/ex1/ex1data1.txt
load ./machine-learning-ex1/ex1/ex1data2.txt

disp(who)  % R: ls() includes ex1data1, ex1data2;
disp('size ex1data1')
disp(size(ex1data1))
disp('size ex1data2')
disp(size(ex1data2))
          
save ./hello.mat v; % binary
save ./hello.txt v -ascii % human-readable

clear % clear workspace - R: rm(list=ls())

load hello
disp(who)


A = [1 2; 3 4; 5 6];
A(3, 2)     % 6
A(2, :)     % 3 4
A(:, 2)     % 2; 4; 6
A([1 3], :) % 1 2; 5 6;
A(:, 2) = [10; 11; 12];
disp(A)     % [1 10; 3 11; 5 12]

A = [A, [100; 101; 102]] % R: cbind();
disp(A)

disp(A(:)) % Py: A.ravel() -- returns single col vec

A = [1 2; 3 4; 5 6];
B = [7 8; 9 0; 1 2];
C = [A B] % R: cbind(A, B);
D = [A; B] % R: rbind(A, B);


  

  
                        
