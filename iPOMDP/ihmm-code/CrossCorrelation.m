function [ C ] = CrossCorrelation( S, l )
%CROSSCORRELATION

if nargin > 1
    L = l;
else
    L = length(S);
end
T = length(S{1}.S);
C = zeros(T);

for i=1:L
    B = unique(S{i}.S);
    for b=1:length(B)
        C(S{i}.S == B(b), S{i}.S == B(b)) = C(S{i}.S == B(b), S{i}.S == B(b)) + 1/L;
    end
end