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

