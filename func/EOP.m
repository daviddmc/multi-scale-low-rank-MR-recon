function [ E, ET ] = EOP( mask, smap )

[nx, ny, nc] = size(smap);
smap = reshape(smap, [nx, ny, 1, nc]);

E = @(x) op(x, smap, mask);
ET = @(x) adj(x, smap, mask);

end

function res = op(X, smap, mask)
    
res = X .* smap;
res = fft2c(res);
res = res .* mask;

end

function res = adj(X, smap, mask)

fac = sum(abs((smap)).^2, 4);
res = X .* mask;
res = ifft2c(res);
res = res .* conj(smap);
res = sum(res, 4) ./ fac;

end
