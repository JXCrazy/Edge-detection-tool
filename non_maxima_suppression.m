function maxima_pixel=non_maxima_suppression(Px,Py)
%Px---the partial derivatives along x axis
%Py---the partial derivatives along y axis
maxima_pixel=zeros(size(Px));
Mag_grad=sqrt(Px.^2+Py.^2);
[M,N]=size(Px);
for m=2:M-1
    for n=2:N-1
      a=Mag_grad(m,n);
      if a~=0
        zx=Px(m,n)/a;zy=Py(m,n)/a;
        if abs(zy)<=sin(pi/4)&(zx*zy>=0)
            b=Mag_grad(m+1,n)*(1-zy/zx)+Mag_grad(m+1,n+1)*(zy/zx);
            c=Mag_grad(m-1,n)*(1-zy/zx)+Mag_grad(m-1,n-1)*(zy/zx);
        end
        if abs(zy)>sin(pi/4)&(zx*zy>=0)
            b=Mag_grad(m,n+1)*(1-zx/zy)+Mag_grad(m+1,n+1)*(zx/zy);
            c=Mag_grad(m,n-1)*(1-zx/zy)+Mag_grad(m-1,n-1)*(zx/zy);
        end
        if abs(zy)>=sin(pi/4)&(zx*zy<=0)
            b=Mag_grad(m,n+1)*(1+zx/zy)+Mag_grad(m-1,n+1)*(-zx/zy);
            c=Mag_grad(m,n-1)*(1+zx/zy)+Mag_grad(m+1,n-1)*(-zx/zy);
        end
        if abs(zy)<sin(pi/4)&(zx*zy<=0)
            b=Mag_grad(m-1,n+1)*(-zy/zx)+Mag_grad(m-1,n)*(1+zy/zx);
            c=Mag_grad(m+1,n-1)*(-zy/zx)+Mag_grad(m+1,n)*(1+zy/zx);
        end
        if a>max(b,c)
        maxima_pixel(m,n)=1;
        end
      end  
    end
end

            
            
        