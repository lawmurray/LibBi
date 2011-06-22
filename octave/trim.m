function a = trim (b)
    a = zeros(size(b));
    for i = 1:length (b)
        a(i) = str2num(sprintf('%.3e', b(i)));
    end
end
