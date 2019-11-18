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



% Computing on Data-------------------------------------------------------------
B = [11 12; 13 14; 15 16];
C = [1 1; 2 2];

AC = A * C % mat mult;
disp(AC)

D = A .* B % element-wise;
disp(D)

A2 = A .^2 % elemen-wise squaring;
disp(A2)

v = [1; 2; 3];
1 ./ v % 1; 1/2; 1/3

log(v) % element-wise log

A' % A transpose;
B'

a = [1 15 2 0.5];
[val, ind] = max(a) % val=15 ind = 2;

max(A) % col-wise, so: 5 6;

a < 3       % 1 0 1 1;
find(a < 3) % 1 3 4;

A = magic(3);
[r, c] = find(A >= 7) % r (rows) = 1 3 2, c (cols) = 1 2 3 (eg, 1,1 3,2 2,3);

sum(a)  % 18.5
prod(a) % 15

max(rand(3), rand(3)) % pairwise max of two rand 3x3 mats
max(A, [], 1) % colwise maxes: 8 9 7
max(A)        % same           8 9 7
max(A, [], 2) % rowwise: 8 7 9
max(max(A))   % 9
max(A(:))     % 9

A = magic(9);
sum(A, 1) % colwise sums
sum(A, 2) % rowwise

% sum diagonals
sum(sum(A .* eye(9)))
sum(sum(A .* flipud(eye(9))))

A = magic(3);
Ainv = pinv(A); % "pseudo"-inverse
A * Ainv % I3




  
  



 
  
  
  
  
  
                        
