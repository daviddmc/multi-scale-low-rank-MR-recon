function x = ifft2c( X )
%{
X = ifftshift(x, 2);
X = ifftshift(X, 1);
X = ifft2(X);
X = X * sqrt(size(x, 1) * size(x, 2));
%}
x=fftshift(fft(fftshift(X,1),[],1),1)/sqrt(size(X,1));
x=fftshift(fft(fftshift(x,2),[],2),2)/sqrt(size(X,2));


end

