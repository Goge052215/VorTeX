sum = 0;

for n = 0: 50
    sum = sum + (-1)^n * (2*n+1);
end

disp(sum)