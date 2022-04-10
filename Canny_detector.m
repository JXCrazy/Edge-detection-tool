function edge_map=CFAR_Canny_edge_detector(X,cgma,cent,nst,Pf)
%EDGE Find edges in intensity image.
%   Edge_map takes an intensity or a binary image I as its input, and returns a 
%   binary image BW of the same size as I, with 1's where the function 
%   finds edges in I and 0's elsewhere.
%   X---imput image;
%   cgma---the scale of Gaussian kernel;
%   cent---percent number of determining the upper threshold
%   nst----noise standard devistion
%   pf----false alarm probability
%----------------------------------------------------------
% Transform to a double precision intensity image if necessary
if ~isa(X, 'double') 
  X= im2double(X);
end
[M,N]=size(X);%M-the row number of the image and N-the column number of the image
% image extension
Y=[X(20:-1:2,:);X;X(M-1:-1:M-19,:)];
Y=[Y(:,20:-1:2),Y,Y(:,N-1:-1:N-19)];
%calculate two partial derivative filters
BB0=anisotropic_Directional_derivative_filter(cgma,2,1);
filter_x=BB0(:,:,1);
filter_y=BB0(:,:,2);
%--------------estimate nosie statistics---------------------------
        if nst~=0
            x=randn(512,512);
            xx=sqrt((filter2(filter_x,x,'same')).^2+(filter2(filter_y,x,'same')).^2);
            xx=sort(reshape(xx,512^2,1));
            unit_noise_threshold=xx(fix(512^2*(1-Pf)));
        else
            unit_noise_threshold=0;
        end    
        clear x xx
%----------calculate partial derviatives--------
        Px=filter2(filter_x,Y,'same');
        Py=filter2(filter_y,Y,'same');
%----------calculate gradient magnitudes----------
        Max_grad=sqrt(Px.^2+Py.^2);
%----------non-maxima suppression--------------------------------        
        Maxima_pixel=non_maxima_suppression(Px,Py);
        Maxima_pixel=Maxima_pixel(20:M+38-19,20:N+38-19);
%---------calculate local average variations and average variation
        LAESM=filter2(ones(31,31)/31^2,Max_grad,'same');
        AESM=mean(Max_grad(:));
%----------Contrast Equalization------ 
        Normalization_matrix=AESM+0.5*LAESM(20:M+38-19,20:N+38-19);
        Max_grad=Max_grad(20:M+38-19,20:N+38-19)./Normalization_matrix;
%----------Estimate high_threshold--------
        YY=reshape(Max_grad,M*N,1);
        YY=sort(YY);
        High_threshold=YY(fix(M*N*cent));
        Low_threshold=YY(fix(M*N*0.5));
        clear YY
%-----------lower_threshold designation--------------
        Low_threshold=max(Low_threshold*ones(M,N),unit_noise_threshold*nst./Normalization_matrix);
%-----------extract strong edges and candidate edge pixels-------
        ZZ_high=(Max_grad>=High_threshold).*Maxima_pixel;
        ZZ_low=(Max_grad>=Low_threshold).*Maxima_pixel;
%-----------extract edge map-------------------------   
        e=ZZ_low;
        idxStrong=find(ZZ_high==1);
        rstrong = rem(idxStrong-1, M)+1;
        cstrong = floor((idxStrong-1)/M)+1;
        e = bwselect(e, cstrong, rstrong,8);
        e = bwmorph(e, 'thin',8);
        edge_map=e;