function X = fft2c( x )
%{
X = fft2(x);
X = fftshift(X, 1);
X = fftshift(X, 2);
X = X / sqrt(size(x, 1) * size(x, 2));
%}
X=fftshift(ifft(fftshift(x,1),[],1),1)*sqrt(size(x,1));
X=fftshift(ifft(fftshift(X,2),[],2),2)*sqrt(size(x,2));

end

