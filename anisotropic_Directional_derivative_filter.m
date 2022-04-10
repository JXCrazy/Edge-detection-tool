function BB=anisotropic_Directional_derivative_filter(cgma,P,rou)
    %rou=1/tan(pi/(2*P));
    cta=0:pi/P:pi-pi/P;
    x=-20:1:20;
    y=-20:1:20;
    for k=1:P
        B=kernel_matrix(rou,cta(k));
        for m=1:41
            for n=1:41
                z=[x(m);y(n)];
                BB(m,n,k)=(-rou*(x(m)*cos(cta(k))+y(n)*sin(cta(k)))/cgma)*1/(2*pi*cgma)*exp(-z'*B*z/(2*cgma));
            end
        end
    end    
    
        