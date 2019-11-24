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



% Plotting Data----------------------------------------------------------------
t = [0:0.01:0.98];
y1 = sin(8 * pi * t);
plot(t, y1)
y2 = cos(8 * pi * t);
plot(t, y2) % new plotting window

plot(t, y1)
hold on % use same plotting device
plot(t, y2, 'r')
xlabel('time')
ylabel('height')
legend('sin', 'cos')
title('Sinusoids!')
print -dpng 'myPlot.png'
close % shut plotting device

figure(1)
plot(t, y1)
figure(2) % opens new device; 1 stays open
plot(t, y2)

subplot(1, 2, 1) % like matplotlib subplot(121)
plot(t, y1)
subplot(1, 2, 2)
plot(t, y2)
axis([0.5, 1, -1, 1]) % xlim and ylim

clf % clear figure

A = magic(5);
imagesc(A) % R: image; matplotlib imshow();

imagesc(A), colorbar, colormap gray;



% Control flow-----------------------------------------------------------------
v = zeros(10, 1);

for i = 1:10,
  v(i) = 2^i
end;

disp(v)

inds = 1:10;
for i = inds,
  disp(i)
end;

% break and continue work as ususal

i = 1;
while i <= 5,
  v(1) = 100 + i;
  i = i + 1;
end;
disp(v)

i = 1;
while true,
  v(i) = 1000 - i;
  i = i + 1;
  if i == 6,
    break;
  end;
end;

x = 2;
if x == 1,
  disp('1');
elseif x == 2,
  disp('2');
else
  disp('not 1 or 2');
end;


function y = squareThisNumber(x) % y defines the return value
  y = x^2;

disp(squareThisNumber(5))


function [y1, y2] = squareAndCubeThisNumber(x)
  y1 = x^2;
  y2 = x^3;

X = [1 1; 1 2; 1 3];
y = [1; 2; 3];
theta = [0; 1]; % params: intercept slope

function J = costFunction(X, y, theta)
  m = size(X, 1);
  preds = X * theta;
  sse = sum((preds - y).^2);
  J = 1 / (2*m) * sse;

j = costFunction(X, y, theta);
disp(j) % 0 (scaled err)

theta = [1; 0];
j = costFunction(X, y, theta);
disp(j) % 0.333
