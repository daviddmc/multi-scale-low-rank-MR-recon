function ress = mtimes(a,bb)

 if a.adjoint,
     % Multicoil non-Cartesian k-space to Cartesian image domain
     % nufft for each coil and time point
     for tt=1:size(bb,4),
     for ch=1:size(bb,3),
         b = bb(:,:,ch,tt).*a.w(:,:,tt);
         res(:,:,ch,tt) = reshape(nufft_adj(b(:),a.st{tt})/sqrt(prod(a.imSize)),a.imSize(1),a.imSize(2));
     end
     end
     % compensate for undersampling factor
     res=res*size(a.b1,1)*pi/2/size(a.w,2);     
     % coil combination for each time point
     for tt=1:size(bb,4),
         ress(:,:,tt)=sum(res(:,:,:,tt).*conj(a.b1),3)./sum(abs((a.b1)).^2,3); %#ok<AGROW>
     end
 else
     % Cartesian image to multicoil non-Cartesian k-space 
     for tt=1:size(bb,3),
     for ch=1:size(a.b1,3),
        res=bb(:,:,tt).*a.b1(:,:,ch); %#ok<AGROW>
        ress(:,:,ch,tt) = reshape(nufft(res,a.st{tt})/sqrt(prod(a.imSize)),a.dataSize(1),a.dataSize(2)).*a.w(:,:,tt);
     end
     end
 end
            
