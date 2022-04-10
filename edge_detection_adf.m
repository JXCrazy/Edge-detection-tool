function edge_map=edge_detection_adf(X,cgma,P,rou,cent,nst,Pf)
    %X----input gray-level image
    %cgma---the scale of the anisotropic Gaussian kernels
    %P---the number of anisotropic directional derivative filters
    %cent---percent to determine the high_threshold, generally, is between
           %0.6<=cent<=0.95 and low-threshold equals to t*high_threshold
           %0.4<=t<=0.6
    %nst----standarad deivation of Gaussian white noise that corrupts the
    %image;
    %-----------calculate anisotropic directional derivative filters
         BB=anisotropic_Directional_derivative_filter(cgma,P,rou);
    %-----------calculate the two isotropic partial derivative filters----
         BB0=anisotropic_Directional_derivative_filter(cgma/rou,2,1);
         % All filters share the same spatial support[-10,10]*[-10,10];
    %--------------estimate nosie statistics---------------------------
        if nst~=0
            x=randn(512,512);
            Px=filter2(BB0(:,:,1),x,'same');
            Py=filter2(BB0(:,:,2),x,'same');
            ESM0=sqrt(Px.^2+Py.^2);
            xx=abs(filter2(BB(:,:,1),x,'same'));
            for p=2:P
                xx=max(xx,abs(filter2(BB(:,:,p),x,'same')));
            end
            xx=sort(reshape(sqrt(xx.*ESM0),512^2,1));
            unit_noise_threshold=xx(fix(512^2*(1-Pf)));
        else
            unit_noise_threshold=0;
        end    
        % The false alarm probability is specified as 0.01
            clear x ESM0 Px Py xx
    %-----------image extnesion-------------------------------------
        [M,N]=size(X);
        Y=[X(20:-1:2,:);X;X(M-1:-1:M-19,:)];
        Y=[Y(:,20:-1:2),Y,Y(:,N-1:-1:N-19)];
        % image is symmetric extended to size (M+38)*(N+38);
    %-----------calculate the conbining edge strength map-----------
    %calculate two paritial derivatives, gradient mudulus
        Px=filter2(BB0(:,:,1),Y,'same');
        Py=filter2(BB0(:,:,2),Y,'same');
        ESM0=sqrt(Px.^2+Py.^2);
    %non_maxima suppression
        Max_index=non_maxima_suppression(Px,Py);
    %Maximal amplitudes of the anisotropic Directional derivatives
        ESM1=abs(filter2(BB(:,:,1),Y,'same'));
        for p=2:P
            ESM1=max(ESM1, abs(filter2(BB(:,:,p),Y,'same')));
        end    
    %Combining edge strength map
        ESM=sqrt(ESM1.*ESM0);
    %Calculate the average variation and locally average varitions of image
        LAESM=filter2(ones(31,31)/31^2,ESM0,'same');
        AESM=mean(ESM0(:));
    %Cut out the extended regions of the image    
        Max_index=Max_index(20:M+38-19,20:N+38-19);
        ESM=ESM(20:M+38-19,20:N+38-19);
    %Contrast equilization
        Enhancement_matrix=AESM+0.5*LAESM(20:M+38-19,20:N+38-19);
        ESM=ESM./Enhancement_matrix;
    %----Estimate upper_threshold--------
        YY=reshape(ESM,1,M*N);
        YY=sort(YY);
        Upper_threshold=YY(fix(cent*M*N));
        Lower_threshold=YY(fix(0.5*M*N));
        clear YY
    %----Estimate lower-threshold--------------
        Lower_threshold=max(Lower_threshold,unit_noise_threshold*nst./Enhancement_matrix);
        clear Enhancement_matrix
    %-----extract the strong edge pixels and candidate edge pixels
        ZZ_strong=(ESM>=Upper_threshold).*Max_index;
        ZZ_possible=(ESM>=Lower_threshold).*Max_index;
    %-----extract edge map   
        e=ZZ_possible;
        idxStrong=find(ZZ_strong==1);
        rstrong = rem(idxStrong-1, M)+1;
        cstrong = floor((idxStrong-1)/M)+1;
        e = bwselect(e, cstrong, rstrong,8);
        e = bwmorph(e, 'thin',8);
        edge_map=e;
               
               
               